/**
 * Copyright 2019/2020 Andreas Ley
 * Written for Digital Image Processing of TU Berlin
 * For internal use in the course only
 */

#include "Dip6.h"

#include "Network.h"
#include "NetworkArchitectures.h"
#include "TaskScheduler.h"
#include "StopWatch.h"

#include <opencv2/opencv.hpp>

#include <iostream>
#include <iomanip>
#include <random>




// usage: path to image in argv[1], SNR in argv[2], stddev of Gaussian blur in argv[3]
// main function. Loads the image, calls test and processing routines, records processing times
int main(int argc, char** argv) {
    
    if (argc < 4) {
        std::cout << "Usage: ./application snapshotFile inputImage outputImage" << std::endl;
        std::cout << "The snapshot file is the xxxxxxxx_snapshot.bin file produced by ./trainBig" << std::endl;
        return 0;
    }
    
    TaskScheduler::Init(std::thread::hardware_concurrency());

    std::mt19937 rng(0xC0FFEE); // Use fixed seed to make things reproducible.

    //cv::Mat inputImage = cv::imread("../buildRelease/lena.jpg", cv::IMREAD_COLOR);
    cv::Mat inputImage = cv::imread(argv[2], cv::IMREAD_COLOR);
    
    
    unsigned cropSize = 128;
    unsigned batchSize = 16;
    
    dip6::Network network = dip6::buildBigNetwork(rng);
    std::cout << "Loading snapshot..." << std::flush;
    network.restoreSnapshot(argv[1]);
    std::cout << "done" << std::endl;
    

    dip6::Tensor inputTensor;
    inputTensor.allocate(cropSize, cropSize, 3, batchSize);
    unsigned outputCropSize;
    {
        inputTensor.setZero();
        const dip6::Tensor &outputTensor = network.forward(inputTensor);
        outputCropSize = outputTensor.getSize(0);
    }
    
    int shift = (cropSize*3 - outputCropSize)/2;
    shift = std::round(shift/3.0f);
    
    cv::Mat outputImage(inputImage.rows*3, inputImage.cols*3, CV_8UC3, cv::Scalar(0, 0, 0));
    
    struct Crop {
        unsigned x, y;
    };
    
    std::vector<Crop> crops;
    
    unsigned stride = outputCropSize - outputCropSize % 3;
    
    for (unsigned y = 0; y < outputImage.rows; y += stride)
        for (unsigned x = 0; x < outputImage.cols; x += stride) {
            crops.push_back({x, y});
        }
            
    
    
    unsigned numBatches = (crops.size()+batchSize-1)/batchSize;
    for (unsigned i = 0; i < numBatches; i++) {
        std::cout << "Running batch " << i << " of " << numBatches << std::endl;
        for (unsigned j = 0; j < batchSize; j++) {
            Crop &crop = crops[(i*batchSize + j) % crops.size()];
            
            for (unsigned y = 0; y < cropSize; y++)
                for (unsigned x = 0; x < cropSize; x++) {
                    int in_y = (int)crop.y/3-(int)shift + (int)y;
                    int in_x = (int)crop.x/3-(int)shift + (int)x;
           
                    in_y = std::min<int>(std::max<int>(in_y, 0), inputImage.rows-1);
                    in_x = std::min<int>(std::max<int>(in_x, 0), inputImage.cols-1);
                    
                    cv::Vec3b color = inputImage.at<cv::Vec3b>(in_y, in_x);
                    inputTensor(y, x, 0, j) = color[0] / 127.0f - 1.0f;
                    inputTensor(y, x, 1, j) = color[1] / 127.0f - 1.0f;
                    inputTensor(y, x, 2, j) = color[2] / 127.0f - 1.0f;
                }
        }
        const dip6::Tensor &outputTensor = network.forward(inputTensor);
        for (unsigned j = 0; j < batchSize; j++) {
            if (i*batchSize + j >= crops.size()) break;
            
            Crop &crop = crops[i*batchSize + j];
            
            for (unsigned y = 0; y < stride; y++)
                for (unsigned x = 0; x < stride; x++) {
                    
                    int out_y = crop.y + y;
                    int out_x = crop.x + x;
                    
                    if ((out_y >= outputImage.rows) || (out_x >= outputImage.cols))
                        continue;
                    
                    cv::Vec3b &outputPixel = outputImage.at<cv::Vec3b>(out_y, out_x);
                    
                    outputPixel[0] = std::min<int>(std::max<int>(outputTensor(y, x, 0, j) * 127.0f + 127.0f, 0), 255);
                    outputPixel[1] = std::min<int>(std::max<int>(outputTensor(y, x, 1, j) * 127.0f + 127.0f, 0), 255);
                    outputPixel[2] = std::min<int>(std::max<int>(outputTensor(y, x, 2, j) * 127.0f + 127.0f, 0), 255);
                }
        }
    }
    std::cout << "Writing output image" << std::endl;
    cv::imwrite(argv[3], outputImage);

    return 0;
} 
