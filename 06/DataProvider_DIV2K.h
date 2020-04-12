/**
 * Copyright 2019/2020 Andreas Ley
 * Written for Digital Image Processing of TU Berlin
 * For internal use in the course only
 */

#ifndef DATAPROVIDER_DIV2K_H
#define DATAPROVIDER_DIV2K_H

#include "Dip6.h"
#include "Tensor.h"

#include <opencv2/opencv.hpp>

#include <cstdint>
#include <random>
#include <string>

#include <fstream>

#include <thread>
#include <mutex>
#include <condition_variable>

namespace dip6 {
    
namespace data {
    
class FileReader {
    public:
        FileReader(const std::string &filename, unsigned prefetchSlots);
        ~FileReader();
        
        void waitPrefetchingDone();
        void seek(std::size_t pos);
        void read(void *dst, std::size_t count);
        void read(void *dst, std::size_t pos, std::size_t count);
        void prefetch(unsigned slot, std::size_t pos, std::size_t count);
        void startPrefetching();
    protected:
        std::mutex m_dataReadingMutex;
        std::thread m_dataReadingThread;
        std::condition_variable m_dataReadingCondition;
        
        struct Buffer {
            std::size_t fileOffset;
            std::vector<unsigned char> data;
        };
        std::vector<Buffer> m_prefetchSlots;
        bool m_dataRead = true;
        bool m_shutdown = false;

        std::fstream m_file;
        
        void dataReadingThread();
};
    

class DIV2K : public DataProvider
{
    public:
        DIV2K(const std::string &dbFilename, unsigned batchSize, std::size_t randomSeed, bool shuffle = true);
        
        void reset();
        bool fetchMinibatch(Tensor &input, Tensor &desiredOutput);
        
        unsigned getNumTiles() const { return m_tiles.size(); }
    protected:
        std::mt19937 m_rng;
        unsigned m_batchSize;
        
        unsigned m_tileWidth = ~0u; 
        unsigned m_tileHeight = ~0u;
        unsigned m_downsamplingFactor = 3;
        
        unsigned m_nextTileIndex = 0;
        
        bool m_shuffle;
        
        std::vector<unsigned> m_shuffledTileIndices;
        struct Tile {
            std::uint64_t offset;
            std::uint64_t size;
        };
        std::vector<Tile> m_tiles;
        FileReader m_fileReader;
        
        struct InstanceBuffer {
            std::vector<unsigned char> dataBuffer;
            cv::Mat_<cv::Vec3b> tile;
            cv::Mat_<cv::Vec3f> desiredOutput;
            cv::Mat_<cv::Vec3f> input;
            std::mt19937 instanceRng;
        };
        
        std::vector<InstanceBuffer> m_instanceBuffer;
        
        void process(cv::Mat_<cv::Vec3b> &tile, cv::Mat_<cv::Vec3f> &desiredOutput, cv::Mat_<cv::Vec3f> &input, std::mt19937 &instanceRng);
};

}

}

#endif // DATAPROVIDER_DIV2K_H
