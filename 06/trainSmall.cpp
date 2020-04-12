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


#include <iostream>
#include <iomanip>
#include <random>



class RngProvider : public dip6::DataProvider
{
    public:
        RngProvider(unsigned w, unsigned h, unsigned c, unsigned batchSize) : m_width(w), m_height(h), m_channels(c), m_batchSize(batchSize) { }
        
        void reset() override { }
        bool fetchMinibatch(dip6::Tensor &input, dip6::Tensor &desiredOutput) override {
            input.allocate(m_height, m_width, m_channels, m_batchSize);
            desiredOutput.allocate(m_height, m_width, m_channels, m_batchSize);
            for (unsigned i = 0; i < input.getTotalSize(); i++)
                input[i] = desiredOutput[i] = m_dist(m_rng);
            return false;
        }
    protected:
        unsigned m_width;
        unsigned m_height;
        unsigned m_channels;
        unsigned m_batchSize;
        
        std::normal_distribution<float> m_dist;
        std::mt19937 m_rng;
};

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

    StopWatch totalRuntime;

    TaskScheduler::Init(std::thread::hardware_concurrency());


    std::mt19937 rng(0xC0FFEE); // Use fixed seed to make things reproducible.

    dip6::Network network = dip6::buildSmallNetwork(rng);
    RngProvider rngProvider(16, 16, 3, 16);
    dip6::MSELoss mseLoss;

    std::cout << "=== Small network benchmark ===" << std::endl;
    std::cout << "The speed of the small network (using the reference implementations of the convlution) is not super critical but should be < 10ms in total." << std::endl << std::endl;
    
    {
        dip6::Tensor inputData, desiredOutputData;
        rngProvider.fetchMinibatch(inputData, desiredOutputData);
        
        const dip6::Tensor &output = network.forward(inputData);
        dip6::Tensor outputGradTensor;
        outputGradTensor.allocateLike(output);
        
        
        network.benchmarkForward(inputData);
        network.benchmarkBackward(inputData, outputGradTensor);
    }

    float avgTrainingError = 0.0f;
    
    auto runAvg = [](unsigned iter, float &f, float inc) {
        if (iter == 0) f = inc; else f = f * 0.99f + inc * 0.01f;
    };
    
    std::cout << "=== Identity learning test ===" << std::endl;
    std::cout << "The small network is now trying to learn the identity function (input = output)." << std::endl;
    std::cout << "The loss should go down to virtually zero. If it doesn't, something is not right." << std::endl << std::endl;
    std::cout << "A loss close to 1.0 probably means the network always returns zero." << std::endl << std::endl;
    
    unsigned epoch = 0;
    dip6::Tensor inputData, desiredOutputData, outputGradTensor;
    const unsigned totalIterations = 10'000;
    for (unsigned i = 0; i < totalIterations; i++) {
        rngProvider.fetchMinibatch(inputData, desiredOutputData);

        const dip6::Tensor &output = network.forward(inputData);
        float loss = mseLoss.computeLoss(output, desiredOutputData, outputGradTensor);

        runAvg(i, avgTrainingError, loss);
        
        network.backward(inputData, outputGradTensor);
        network.updateParameters(0.1f);


        if (i % 100 == 0) {
            std::cout << "\rIter: " << i << " Epoch: " << epoch << " Loss: " << avgTrainingError;
            
            std::cout << " | Runtime: ";
            formatSeconds(totalRuntime.getElapsedSeconds(), std::cout);
            std::cout << " | Time remaining: ";
            formatSeconds((totalIterations-(i+1)) * totalRuntime.getElapsedSeconds() / (i+1), std::cout);
            std::cout << "     " << std::flush;
        }
    }
    std::cout << std::endl;

    return 0;
} 
