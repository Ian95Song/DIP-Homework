/**
 * Copyright 2019/2020 Andreas Ley
 * Written for Digital Image Processing of TU Berlin
 * For internal use in the course only
 */

#include "DataProvider_DIV2K.h"
#include "TaskScheduler.h"


#include <stdexcept>
#include <algorithm>

#ifdef DIV2K_BUILD_WITH_PREFETCHER
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#else
#include <iostream>
#endif

namespace dip6 {
    
namespace data {


FileReader::FileReader(const std::string &filename, unsigned prefetchSlots)
{
    m_file.rdbuf()->pubsetbuf(0, 0);
    m_file.open(filename.c_str(), std::fstream::in | std::fstream::binary);
    if (!m_file)
        throw std::runtime_error("Could not open file!");
    
    m_prefetchSlots.resize(prefetchSlots);
    for (unsigned i = 0; i < m_prefetchSlots.size(); i++)
        m_prefetchSlots[i].fileOffset = ~0ul;
    
    m_dataReadingThread = std::thread(&FileReader::dataReadingThread, this);
}


FileReader::~FileReader()
{
    {
        std::lock_guard<std::mutex> lock(m_dataReadingMutex);
        m_shutdown = true;
        m_dataReadingCondition.notify_all();
    }   
    m_dataReadingThread.join();
}

void FileReader::waitPrefetchingDone()
{
//    if (!m_dataRead)
//        std::cout << "Reading data not yet done. Hdd can't keep up" << std::endl;
    std::unique_lock<std::mutex> lock(m_dataReadingMutex);
    while (!m_dataRead)
        m_dataReadingCondition.wait(lock);
}

void FileReader::seek(std::size_t pos)
{
    m_file.seekg(pos);
}

void FileReader::read(void *dst, std::size_t count)
{
    m_file.read((char*)dst, count);
}

void FileReader::read(void *dst, std::size_t pos, std::size_t count)
{
    for (auto &b : m_prefetchSlots) {
        if ((b.fileOffset == pos) && (b.data.size() == count)) {
            memcpy(dst, b.data.data(), count);
            return;
        }
    }
//    std::cout << "Data not prefetched!" << std::endl;
    seek(pos);
    read(dst, count);
}

void FileReader::prefetch(unsigned slot, std::size_t pos, std::size_t count)
{
    m_prefetchSlots[slot].fileOffset = pos;
    m_prefetchSlots[slot].data.resize(count);
}

void FileReader::startPrefetching()
{
    std::unique_lock<std::mutex> lock(m_dataReadingMutex);
    m_dataRead = false;
    m_dataReadingCondition.notify_all();
}

void FileReader::dataReadingThread()
{
    while (true) {
        {
            std::unique_lock<std::mutex> lock(m_dataReadingMutex);
            while (m_dataRead) {
                if (m_shutdown) return;
                m_dataReadingCondition.wait(lock);
            }
        }
        for (auto &b : m_prefetchSlots) {
            if (b.fileOffset != ~0ul) {
                seek(b.fileOffset);
                read(b.data.data(), b.data.size());
            }
        }
        {
            std::unique_lock<std::mutex> lock(m_dataReadingMutex);
            m_dataRead = true;
            m_dataReadingCondition.notify_all();
        }
    }
}



DIV2K::DIV2K(const std::string &dbFilename, unsigned batchSize, std::size_t randomSeed, bool shuffle) : m_rng(randomSeed), m_batchSize(batchSize), m_fileReader(dbFilename, batchSize), m_shuffle(shuffle)
{
    std::uint64_t numTiles;
    m_fileReader.read(&numTiles, sizeof(numTiles));
    m_tiles.resize(numTiles);
    m_fileReader.read(m_tiles.data(), m_tiles.size() * sizeof(Tile));
    
    std::vector<unsigned char> data;
    data.resize(m_tiles[0].size);
    m_fileReader.read(data.data(), data.size());
    cv::Mat img = cv::imdecode(cv::_InputArray((const char *)data.data(), data.size()), 0);
    if (img.empty())
        throw std::runtime_error("Could not read first image from db! File corrupted?");
    
    m_tileWidth = img.cols;
    m_tileHeight = img.rows;
    
    reset();
}

void DIV2K::reset()
{
    m_shuffledTileIndices.resize(m_tiles.size());
    for (unsigned i = 0; i < m_shuffledTileIndices.size(); i++)
        m_shuffledTileIndices[i] = i;
    
    if (m_shuffle)
        std::shuffle(m_shuffledTileIndices.begin(), m_shuffledTileIndices.end(), m_rng);
    m_nextTileIndex = 0;
}

bool DIV2K::fetchMinibatch(Tensor &input, Tensor &desiredOutput)
{
    if (m_batchSize != m_instanceBuffer.size()) {
        m_instanceBuffer.resize(m_batchSize);
        for (unsigned instance = 0; instance < m_batchSize; instance++) {
            m_instanceBuffer[instance].instanceRng.seed(m_rng());
        
            m_instanceBuffer[instance].desiredOutput.create(m_tileHeight, m_tileWidth);
            m_instanceBuffer[instance].input.create(m_tileHeight / m_downsamplingFactor, m_tileWidth / m_downsamplingFactor);
        }
    }

    input.allocate(
        m_tileHeight / m_downsamplingFactor,
        m_tileWidth / m_downsamplingFactor,
        3,
        m_batchSize
    );
    desiredOutput.allocate(
        m_tileHeight,
        m_tileWidth,
        3,
        m_batchSize
    );

    m_fileReader.waitPrefetchingDone();

    TaskGroup decodeGroup;
    for (unsigned instance = 0; instance < m_batchSize; instance++) {
        unsigned tileIdx = m_shuffledTileIndices[(m_nextTileIndex+instance)%m_shuffledTileIndices.size()];
        
        auto &dataBuffer = m_instanceBuffer[instance].dataBuffer;
        
        dataBuffer.resize(m_tiles[tileIdx].size);
        m_fileReader.read(dataBuffer.data(), m_tiles[tileIdx].offset, dataBuffer.size());
        
        decodeGroup.add([instance, this]{
            auto &dataBuffer = m_instanceBuffer[instance].dataBuffer;
            auto &tile = m_instanceBuffer[instance].tile;
            cv::imdecode(cv::_InputArray((const char *)dataBuffer.data(), dataBuffer.size()), cv::IMREAD_COLOR, &tile);
            
            if (tile.empty())
                throw std::runtime_error("Could not read tile from db! File corrupted?");
            
            process(
                tile, 
                m_instanceBuffer[instance].desiredOutput, 
                m_instanceBuffer[instance].input,
                m_instanceBuffer[instance].instanceRng
            );
        });
    }
    
    // prefetch
    for (unsigned instance = 0; instance < m_batchSize; instance++) {
        unsigned tileIdx = m_shuffledTileIndices[(m_nextTileIndex+m_batchSize+instance)%m_shuffledTileIndices.size()];
        m_fileReader.prefetch(instance, m_tiles[tileIdx].offset, m_tiles[tileIdx].size);
    }    
    m_fileReader.startPrefetching();


    decodeGroup.waitFor();

    
    // Single threaded data interleaving to avoid cash thrashing
    for (unsigned instance = 0; instance < m_batchSize; instance++) {
        auto &instDesiredOutput = m_instanceBuffer[instance].desiredOutput;
        auto &instInput = m_instanceBuffer[instance].input;

        for (unsigned y = 0; y < m_tileHeight; y++) {
            for (unsigned x = 0; x < m_tileWidth; x++) {
                desiredOutput(y, x, 0, instance) = instDesiredOutput(y, x)[0];
                desiredOutput(y, x, 1, instance) = instDesiredOutput(y, x)[1];
                desiredOutput(y, x, 2, instance) = instDesiredOutput(y, x)[2];
            }
        }
        for (unsigned y = 0; y < m_tileHeight / m_downsamplingFactor; y++) {
            for (unsigned x = 0; x < m_tileWidth / m_downsamplingFactor; x++) {
                input(y, x, 0, instance) = instInput(y, x)[0];
                input(y, x, 1, instance) = instInput(y, x)[1];
                input(y, x, 2, instance) = instInput(y, x)[2];
            }
        }
    }


    m_nextTileIndex += m_batchSize;
    return m_nextTileIndex >= m_shuffledTileIndices.size();
}

void DIV2K::process(cv::Mat_<cv::Vec3b> &tile, cv::Mat_<cv::Vec3f> &desiredOutput, cv::Mat_<cv::Vec3f> &input, std::mt19937 &instanceRng)
{
    input.setTo(0.0f);
    
    if ((tile.rows != m_tileHeight) || (tile.cols != m_tileWidth)) {
        throw std::runtime_error("Invalid tile size found in DB!");
    }
    
    for (unsigned y = 0; y < m_tileHeight; y++) {
        for (unsigned x = 0; x < m_tileWidth; x++) {
            auto color = tile(y, x);
            cv::Vec3f colorF(color[0] / 128.0f - 1.0f, color[1] / 128.0f - 1.0f, color[2] / 128.0f - 1.0f);
            
            desiredOutput(y, x) = colorF;
            if ((y/m_downsamplingFactor < input.rows) && (x/m_downsamplingFactor < input.cols))
                input(y/m_downsamplingFactor, x/m_downsamplingFactor) += colorF;
        }
    }

    std::normal_distribution<float> noiseDist(0.0f, 0.01f);
    for (unsigned y = 0; y < m_tileHeight/m_downsamplingFactor; y++) {
        for (unsigned x = 0; x < m_tileWidth/m_downsamplingFactor; x++) {
            auto color = input(y, x);
            color *= 1.0f / (m_downsamplingFactor*m_downsamplingFactor);
            color[0] += noiseDist(instanceRng);
            color[1] += noiseDist(instanceRng);
            color[2] += noiseDist(instanceRng);
            input(y, x) = color;
        }
    }
}



}
}
