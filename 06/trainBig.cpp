/**
 * Copyright 2019/2020 Andreas Ley
 * Written for Digital Image Processing of TU Berlin
 * For internal use in the course only
 */

#include "Dip6.h"

#include "NetworkArchitectures.h"
#include "DataProvider_DIV2K.h"
#include "TaskScheduler.h"
#include "StopWatch.h"


#include <opencv2/opencv.hpp>

#include <iostream>
#include <iomanip>
#include <random>




cv::Mat renderColorTensor(const dip6::Tensor &tensor, float shift = 127.0f, float scale = 127.0f)
{
    if (tensor.getSize(2) != 3)
        throw std::runtime_error("Tensor must have 3 channels!");
    
    unsigned numX = std::ceil(std::sqrt((float)tensor.getSize(3)));
    unsigned numY = tensor.getSize(3) / numX;
    
    unsigned border = 1;
    
    cv::Mat result(border + numY * (border + tensor.getSize(0)), border + numX * (border + tensor.getSize(1)), CV_8UC3, cv::Scalar(0, 0, 0));
    for (unsigned ty = 0; ty < numY; ty++)
        for (unsigned tx = 0; tx < numX; tx++) {
            unsigned instance = tx + ty * numX;
            unsigned offsetX = border + tx * (border + tensor.getSize(1));
            unsigned offsetY = border + ty * (border + tensor.getSize(0));
            for (unsigned y = 0; y < tensor.getSize(0); y++)
                for (unsigned x = 0; x < tensor.getSize(1); x++) 
                    for (unsigned c = 0; c < 3; c++) {
                        result.at<cv::Vec3b>(offsetY + y, offsetX + x)[c] = std::min<int>(std::max<int>(tensor(y, x, c, instance) * scale + shift, 0), 255);
                    }
        }
                    
    
    return result;
}

void formatSeconds(float seconds, std::ostream &stream)
{
    unsigned totalSeconds = seconds;
    unsigned h = totalSeconds / 60 / 60;
    totalSeconds -= h * 60 * 60; 
    unsigned m = totalSeconds / 60;
    totalSeconds -= m * 60; 
    unsigned s = totalSeconds;
    stream << h << ':' << m << ':' << s;
}

// usage: path to image in argv[1], SNR in argv[2], stddev of Gaussian blur in argv[3]
// main function. Loads the image, calls test and processing routines, records processing times
int main(int argc, char** argv) {
    
    if (argc < 2) {
        std::cout << "Usage: ./trainBig path_to_training_data" << std::endl;
        std::cout << "The data path must contain the train_72.db and valid_72.db files." << std::endl;
        return 0;
    }

    StopWatch totalRuntime;

    TaskScheduler::Init(std::thread::hardware_concurrency());
    
    std::string dataPath = argv[1];
    bool dataIsOnSSD = false; // This turns on shuffling of the data


    std::mt19937 rng(0xC0FFEE); // Use fixed seed to make things reproducible.

    dip6::Network network = dip6::buildBigNetwork(rng);
    dip6::MSELoss loss;
    dip6::data::DIV2K trainingData(dataPath+"/train_72.db", 16, rng(), dataIsOnSSD);
    std::cout << "Num training tiles: " << trainingData.getNumTiles() << std::endl;
    dip6::data::DIV2K validationData(dataPath+"/valid_72.db", 16, rng(), dataIsOnSSD);

    std::cout << "=== Big network benchmark ===" << std::endl;
    std::cout << "The speed of the big network (using the optimized implementations of the convolution) is *very* important. It should be < 100ms in total or you will wait forever." << std::endl << std::endl;
    {
        dip6::Tensor inputData, desiredOutputData;
        trainingData.fetchMinibatch(inputData, desiredOutputData);
        
        const dip6::Tensor &output = network.forward(inputData);
        dip6::Tensor outputGradTensor;
        outputGradTensor.allocateLike(output);
        
        network.benchmarkForward(inputData);
        network.benchmarkBackward(inputData, outputGradTensor);
    }
    
    double avgTrainingError = 0.0;
    
    auto runAvg = [](unsigned iter, double &f, float inc) {
        if (iter == 0) f = inc; else f = f * 0.99 + inc * 0.01;
    };
    
    unsigned epoch = 0;
    dip6::Tensor inputData, desiredOutputData, outputGradTensor;
    const unsigned totalIterations = 400'000;
    for (unsigned i = 0; i < totalIterations; i++) {
        if (trainingData.fetchMinibatch(inputData, desiredOutputData)) {
            epoch++;
            trainingData.reset();
        }

        const dip6::Tensor &output = network.forward(inputData);
        float l = loss.computeLoss(output, desiredOutputData, outputGradTensor);
        runAvg(i, avgTrainingError, l);


        network.backward(inputData, outputGradTensor);

        if (i < totalIterations/2)
            network.updateParameters(0.001f);
        else
            network.updateParameters(0.0001f);


        if (i % 100 == 0) {
            std::cout << "\rIter: " << i << " Epoch: " << epoch << " Loss: " << avgTrainingError;
            
            std::cout << " | Runtime: ";
            formatSeconds(totalRuntime.getElapsedSeconds(), std::cout);
            std::cout << " | Time remaining: ";
            formatSeconds((totalIterations-(i+1)) * totalRuntime.getElapsedSeconds() / (i+1), std::cout);
            std::cout << "     " << std::flush;
        }
        if (i % 10'000 == 0) {
            std::stringstream prefix;
            prefix << std::setw(8) << std::setfill('0') << i << "_";
            
            cv::imwrite(prefix.str()+"input.png", renderColorTensor(inputData));
            cv::imwrite(prefix.str()+"output.png", renderColorTensor(output));
            cv::imwrite(prefix.str()+"desiredOutput.png", renderColorTensor(desiredOutputData));
            network.saveSnapshot(prefix.str()+"snapshot.bin");
        }
        if (i % 50'000 == 0) {
            std::cout << std::endl;
            std::cout << "Running validation " << std::flush;
            double valLoss = 0.0;
            validationData.reset();
            unsigned numTiles = 0;
            while (!validationData.fetchMinibatch(inputData, desiredOutputData)) {
                numTiles += inputData.getSize(3);
                if (numTiles % (inputData.getSize(3) * 100) == 0)
                    std::cout << "\rRunning validation " << numTiles << " / " << validationData.getNumTiles() << "    " << std::flush;
                const dip6::Tensor &output = network.forward(inputData);

                float subSum = loss.computeLoss(output, desiredOutputData, outputGradTensor);
                valLoss += subSum * inputData.getSize(3);
            }
            std::cout << "\rIter: " << i << " Validation loss: " << valLoss/numTiles << " (error per tile)" << std::endl;
        }
    }
    std::cout << std::endl;

    return 0;
} 
