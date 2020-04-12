/**
 * Copyright 2019/2020 Andreas Ley
 * Written for Digital Image Processing of TU Berlin
 * For internal use in the course only
 */


#include "Dip6.h"

#include "TaskScheduler.h"

#include <opencv2/opencv.hpp>


#include <random>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <mutex>
#include <iomanip>
#include <sstream>


int main(int argc, char** argv) {
    
    TaskScheduler::Init(std::thread::hardware_concurrency());
    
    
    if (argc != 5) {
        std::cout << "Usage: createDB databaseFilename /path/to/images/ startImageIndex numberOfImages" << std::endl;
        return 0;
    }
    
    const char *databaseFilename = argv[1];
    std::string tmpFilename = std::string(databaseFilename)+".tmp";
    const char *imagePath = argv[2];
    unsigned startImageIndex = atoi(argv[3]);
    unsigned numImages = atoi(argv[4]);
    
    const unsigned tileSize = 72;
    const unsigned tileStride = tileSize/2;
    
    std::cout << "Attempting to extract " << tileSize << "^2 pixel tiles with a stride of " << tileStride << " pixels from " << numImages << " images in " << imagePath << " and packing them in " << databaseFilename << std::endl;
    std::cout << "Using " << tmpFilename << " as temporary storage." << std::endl;
    std::vector<std::pair<std::size_t, std::size_t>> tileOffsetSize;
    
    {
        std::fstream tmpFile(tmpFilename.c_str(), std::fstream::out | std::fstream::binary);
        std::size_t tmpFileBytesWritten = 0;
        
        std::mutex mutex;
        
        parallelFor(startImageIndex, startImageIndex+numImages, 1, [&](unsigned imageIndex){
            std::stringstream filename;
            filename << imagePath << std::setw(4) << std::setfill('0') << imageIndex << ".png";
            
            {
                std::lock_guard<std::mutex> lock(mutex);
                std::cout << "Processing " << filename.str() << std::endl;
            }
            
            cv::Mat img = cv::imread(filename.str());
            if (img.empty()) {
                std::lock_guard<std::mutex> lock(mutex);
                std::cout << "Error occured reading " << filename.str() << " Skipping." << std::endl;
                return;
            }
            
            if ((img.cols < tileSize) || (img.rows < tileSize))
                return;
            
            
            std::vector<unsigned char> buffer;
            
            for (unsigned ty = 0; ty+tileSize <= img.rows; ty += tileStride)
                for (unsigned tx = 0; tx+tileSize <= img.cols; tx += tileStride) {
                    
                    cv::Mat tile = img(cv::Rect(tx, ty, tileSize, tileSize));
                    cv::imencode(".jpg", tile, buffer, {cv::IMWRITE_JPEG_QUALITY, 98});

                    {
                        std::lock_guard<std::mutex> lock(mutex);
                        tileOffsetSize.push_back({tmpFileBytesWritten, buffer.size()});
                        tmpFile.write((const char*) buffer.data(), buffer.size());
                        tmpFileBytesWritten += buffer.size();
                    }
                }
            
        });
        
        std::cout << "Extracted " << tileOffsetSize.size() << " tiles." << std::endl;
        std::cout << "Total data size: " << (tmpFileBytesWritten >> 20) << " MB" << std::endl;    
    }
    
    
    {
        std::cout << "Writing (and shuffling) to " << databaseFilename << std::endl;    

        std::mt19937 rng;
        std::vector<std::size_t> randomOrder;
        randomOrder.resize(tileOffsetSize.size());
        for (unsigned i = 0; i < tileOffsetSize.size(); i++)
            randomOrder[i] = i;
        std::shuffle(randomOrder.begin(), randomOrder.end(), rng);
        
        std::vector<std::pair<std::size_t, std::size_t>> dstTileOffsetSize;

        std::fstream databaseFile(databaseFilename, std::fstream::out | std::fstream::binary);
        std::uint64_t totalNumber = tileOffsetSize.size();
        databaseFile.write((const char*) &totalNumber, sizeof(totalNumber));
        std::size_t runningTileOffset = sizeof(std::uint64_t) * (1 + tileOffsetSize.size()*2);
        for (auto idx : randomOrder) {
            auto offsetSize = tileOffsetSize[idx];

            std::uint64_t offset = runningTileOffset;
            databaseFile.write((const char*) &offset, sizeof(offset));
            std::uint64_t size = offsetSize.second;
            databaseFile.write((const char*) &size, sizeof(size));
            
            dstTileOffsetSize.push_back({runningTileOffset, size});
            runningTileOffset += size;
        }

        
        std::vector<unsigned char> buffer;
        
        std::fstream tmpFile;
        tmpFile.rdbuf()->pubsetbuf(0, 0);
        tmpFile.open(tmpFilename.c_str(), std::fstream::in | std::fstream::binary);

        for (auto idx : randomOrder) {
            auto offsetSize = tileOffsetSize[idx];
            buffer.resize(offsetSize.second);
            tmpFile.seekg(offsetSize.first);
            tmpFile.read((char*) buffer.data(), buffer.size());
            databaseFile.write((const char*) buffer.data(), buffer.size());
        }
    }


    return 0;
} 
