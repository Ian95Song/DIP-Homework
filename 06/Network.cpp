/**
 * Copyright 2019/2020 Andreas Ley
 * Written for Digital Image Processing of TU Berlin
 * For internal use in the course only
 */


#include "Network.h"
#include "TaskScheduler.h"
#include "StopWatch.h"
#include "SIMD.h"


#include <typeinfo> 
#include <map> 
#include <string> 
#include <string.h>
#include <stdexcept>
#include <fstream>
#include <iostream>

#include <immintrin.h>

namespace dip6 {

void ParametrizedLayer::updateParameters(float stepsize)
{
    if (m_optimizer != nullptr)
        m_optimizer->performStep(stepsize);
}


void ParametrizedLayer::allocateGradientTensors()
{
    m_parameterGradientTensors.resize(m_parameterTensors.size());
    for (unsigned i = 0; i < m_parameterGradientTensors.size(); i++)
        m_parameterGradientTensors[i].allocateLike(m_parameterTensors[i]);
}

void ParametrizedLayer::saveSnapshot(std::ostream &stream) 
{
    for (const auto &t : m_parameterTensors)
        stream.write((const char*)&t(0, 0, 0, 0), t.getTotalSize() * sizeof(float));
    for (const auto &t : m_parameterGradientTensors)
        stream.write((const char*)&t(0, 0, 0, 0), t.getTotalSize() * sizeof(float));
    if (m_optimizer != nullptr)
        m_optimizer->saveSnapshot(stream);
}

void ParametrizedLayer::restoreSnapshot(std::istream &stream) 
{
    for (auto &t : m_parameterTensors)
        stream.read((char*)&t(0, 0, 0, 0), t.getTotalSize() * sizeof(float));
    for (auto &t : m_parameterGradientTensors)
        stream.read((char*)&t(0, 0, 0, 0), t.getTotalSize() * sizeof(float));
    if (m_optimizer != nullptr)
        m_optimizer->restoreSnapshot(stream);
}



namespace layers {

Conv::Conv(unsigned kernelWidth, unsigned kernelHeight, unsigned inputChannels, unsigned outputChannels)
{
    m_parameterTensors.resize(NUM_PARAMS);
    m_parameterTensors[PARAM_KERNEL].allocate(
        kernelHeight,
        kernelWidth,
        outputChannels,
        inputChannels
    );
    m_parameterTensors[PARAM_KERNEL].setZero();
    m_parameterTensors[PARAM_BIAS].allocate(
        1,
        1,
        outputChannels,
        1
    );
    m_parameterTensors[PARAM_BIAS].setZero();
    allocateGradientTensors();
}

Conv &Conv::initialize(std::mt19937 &rng, float kernelScale, float biasScale)
{
    Tensor &kernel = m_parameterTensors[PARAM_KERNEL];
    Tensor &bias = m_parameterTensors[PARAM_BIAS];
    
    unsigned fanInOut = kernel.getSize(0) * kernel.getSize(1) * (kernel.getSize(2) + kernel.getSize(3))/2;
    
    std::normal_distribution<float> kernelDist(0.0f, kernelScale / std::sqrt(fanInOut / 2.0f));
    for (unsigned y_k = 0; y_k < kernel.getSize(0); y_k++)
        for (unsigned x_k = 0; x_k < kernel.getSize(1); x_k++) 
            for (unsigned c_o = 0; c_o < kernel.getSize(2); c_o++) 
                for (unsigned c_i = 0; c_i < kernel.getSize(3); c_i++) 
                    kernel(y_k, x_k, c_o, c_i) = kernelDist(rng);

    
    std::normal_distribution<float> biasDist(0.0f, biasScale);
    for (unsigned c_o = 0; c_o < bias.getSize(2); c_o++) 
        bias(0, 0, c_o, 0) = biasDist(rng);
    
    return *this;
}

void Conv::resizeOutput(const Tensor &input)
{
    if (input.getSize(2) != m_parameterTensors[PARAM_KERNEL].getSize(3))
        throw std::runtime_error("Kernel input channels do not match input tensor channels!");

    if (input.getSize(0) < m_parameterTensors[PARAM_KERNEL].getSize(0))
        throw std::runtime_error("Input tensor height < kernel height!");
    if (input.getSize(1) < m_parameterTensors[PARAM_KERNEL].getSize(1))
        throw std::runtime_error("Input tensor width < kernel width!");
    
    m_output.allocate(
        input.getSize(0) - m_parameterTensors[PARAM_KERNEL].getSize(0)+1,     // Height
        input.getSize(1) - m_parameterTensors[PARAM_KERNEL].getSize(1)+1,     // Width
        m_parameterTensors[PARAM_KERNEL].getSize(2),                          // Channels
        input.getSize(3)                                                      // Instances
    );
}

void Conv::resizeInputGradients(const Tensor &input, const Tensor &outputGradients)
{
    m_inputGradients.allocateLike(input);
}




Upsample::Upsample(unsigned upsampleX, unsigned upsampleY) : m_upsampleX(upsampleX), m_upsampleY(upsampleY)
{
    
}
        
void Upsample::forward(const Tensor &input)
{
    m_output.allocate(
        input.getSize(0)*m_upsampleY,
        input.getSize(1)*m_upsampleX,
        input.getSize(2),
        input.getSize(3)
    );
#if 0
    for (unsigned y = 0; y < input.getSize(0); y++)
        for (unsigned x = 0; x < input.getSize(1); x++)
            for (unsigned c = 0; c < input.getSize(2); c++)
                for (unsigned instance = 0; instance < input.getSize(3); instance++) {
                    float v = input(y, x, c, instance);
                    for (unsigned y_ = 0; y_ < m_upsampleY; y_++)
                        for (unsigned x_ = 0; x_ < m_upsampleX; x_++)
                            m_output(y*m_upsampleY+y_, x*m_upsampleX+x_, c, instance) = v;
                }
#else
    for (unsigned y = 0; y < input.getSize(0); y++)
        for (unsigned x = 0; x < input.getSize(1); x++)
            for (unsigned c = 0; c < input.getSize(2); c++)
                for (unsigned instance = 0; instance < input.getSize(3); instance+=8) {
                    simd::Vector<8> v(&input(y, x, c, instance));
                    for (unsigned y_ = 0; y_ < m_upsampleY; y_++)
                        for (unsigned x_ = 0; x_ < m_upsampleX; x_++)
                            v.store(&m_output(y*m_upsampleY+y_, x*m_upsampleX+x_, c, instance));
                }
#endif
}

void Upsample::backward(const Tensor &input, const Tensor &outputGradients)
{
    m_inputGradients.allocateLike(input);
#if 0
    for (unsigned y = 0; y < input.getSize(0); y++)
        for (unsigned x = 0; x < input.getSize(1); x++)
            for (unsigned c = 0; c < input.getSize(2); c++)
                for (unsigned instance = 0; instance < input.getSize(3); instance++) {
                    float v = 0.0f;
                    for (unsigned y_ = 0; y_ < m_upsampleY; y_++)
                        for (unsigned x_ = 0; x_ < m_upsampleX; x_++)
                            v += outputGradients(y*m_upsampleY+y_, x*m_upsampleX+x_, c, instance);
                    m_inputGradients(y, x, c, instance) = v;
                }
#else
    for (unsigned y = 0; y < input.getSize(0); y++)
        for (unsigned x = 0; x < input.getSize(1); x++)
            for (unsigned c = 0; c < input.getSize(2); c++)
                for (unsigned instance = 0; instance < input.getSize(3); instance+=8) {
                    simd::Vector<8> v(simd::INIT_ZERO);

                    for (unsigned y_ = 0; y_ < m_upsampleY; y_++)
                        for (unsigned x_ = 0; x_ < m_upsampleX; x_++)
                            v += simd::Vector<8>(&outputGradients(y*m_upsampleY+y_, x*m_upsampleX+x_, c, instance));

                    v.store(&m_inputGradients(y, x, c, instance));
                }
#endif
}


AvgPool::AvgPool(unsigned downsampleX, unsigned downsampleY) : m_downsampleX(downsampleX), m_downsampleY(downsampleY)
{
    
}
        
void AvgPool::forward(const Tensor &input)
{
    m_output.allocate(
        input.getSize(0)/m_downsampleY,
        input.getSize(1)/m_downsampleX,
        input.getSize(2),
        input.getSize(3)
    );
    
    float f = 1.0f / (m_downsampleX*m_downsampleY);
    simd::Scalar factor(&f);
    
    for (unsigned y = 0; y < m_output.getSize(0); y++)
        for (unsigned x = 0; x < m_output.getSize(1); x++)
            for (unsigned c = 0; c < m_output.getSize(2); c++)
                for (unsigned instance = 0; instance < input.getSize(3); instance+=8) {
                    simd::Vector<8> v(simd::INIT_ZERO);

                    for (unsigned y_ = 0; y_ < m_downsampleY; y_++)
                        for (unsigned x_ = 0; x_ < m_downsampleX; x_++)
                            v += simd::Vector<8>(&input(y*m_downsampleY+y_, x*m_downsampleX+x_, c, instance));

                    v *= factor;
                    v.store(&m_output(y, x, c, instance));
                }

}

void AvgPool::backward(const Tensor &input, const Tensor &outputGradients)
{
    m_inputGradients.allocateLike(input);

    float f = 1.0f / (m_downsampleX*m_downsampleY);
    simd::Scalar factor(&f);

    for (unsigned y = 0; y < outputGradients.getSize(0); y++)
        for (unsigned x = 0; x < outputGradients.getSize(1); x++)
            for (unsigned c = 0; c < outputGradients.getSize(2); c++)
                for (unsigned instance = 0; instance < input.getSize(3); instance+=8) {
                    simd::Vector<8> v(&outputGradients(y, x, c, instance));
                    v *= factor;
                    for (unsigned y_ = 0; y_ < m_downsampleY; y_++)
                        for (unsigned x_ = 0; x_ < m_downsampleX; x_++)
                            v.store(&m_inputGradients(y*m_downsampleY+y_, x*m_downsampleX+x_, c, instance));
                }    
}


}


const Tensor &Network::forward(const Tensor &input)
{
    if (m_layer.empty()) throw std::runtime_error("Network doesn't have any layers!");
    m_layer[0]->forward(input);
    for (unsigned i = 1; i < m_layer.size(); i++)
        m_layer[i]->forward(m_layer[i-1]->getLastOutput());
    return m_layer.back()->getLastOutput();
}

void Network::backward(const Tensor &input, const Tensor &outputGradients)
{
    if (m_layer.empty()) throw std::runtime_error("Network doesn't have any layers!");
    if (m_layer.size() == 1) {
        m_layer[0]->backward(input, outputGradients);
    } else {
        unsigned numLayer = m_layer.size();
        m_layer[numLayer-1]->backward(m_layer[numLayer-2]->getLastOutput(), outputGradients);
        for (unsigned i = m_layer.size()-2; i > 0; i--)
            m_layer[i]->backward(m_layer[i-1]->getLastOutput(), m_layer[i+1]->getLastInputGradients());
        m_layer[0]->backward(input, m_layer[1]->getLastInputGradients());
    }
}


void Network::saveSnapshot(std::ostream &stream)
{
    for (auto &layer : m_layer)
        layer->saveSnapshot(stream);
}

void Network::restoreSnapshot(std::istream &stream)
{
    for (auto &layer : m_layer)
        layer->restoreSnapshot(stream);
}

void Network::saveSnapshot(const std::string &filename)
{
    std::fstream file;
    file.open(filename.c_str(), std::fstream::out | std::fstream::binary);
    if (!file)
        throw std::runtime_error("Could not open file!");
    saveSnapshot(file);
}

void Network::restoreSnapshot(const std::string &filename)
{
    std::fstream file;
    file.open(filename.c_str(), std::fstream::in | std::fstream::binary);
    if (!file)
        throw std::runtime_error("Could not open file!");
    restoreSnapshot(file);
}


void Network::benchmarkForward(const Tensor &input)
{
    unsigned numIters = 20;
    
    std::map<std::string, float> timePerType;
    
    if (m_layer.empty()) throw std::runtime_error("Network doesn't have any layers!");
    for (unsigned j = 0; j < numIters; j++) {
        for (unsigned i = 0; i < m_layer.size(); i++) {
            StopWatch stopwatch;
            m_layer[i]->forward(i==0?input:m_layer[i-1]->getLastOutput());
            float time = stopwatch.getElapsedSeconds();
            timePerType[m_layer[i]->layerName()] += time;
        }
    }
    
    float totalTime = 0.0f;
    for (auto &p : timePerType) 
        totalTime += p.second;
    std::cout << "Total forward pass: " << totalTime / numIters * 1000 << " ms" << std::endl;

    for (auto &p : timePerType) {
        std::cout << "Layer type: " << p.first << std::endl;
        std::cout << "  Absolute: " << p.second/numIters*1000 << " ms" << std::endl;
        std::cout << "  Relative: " << p.second/totalTime*100 << " %" << std::endl;
    }
}

void Network::benchmarkBackward(const Tensor &input, const Tensor &outputGradients)
{
    unsigned numIters = 20;
    
    std::map<std::string, float> timePerType;
    
    if (m_layer.empty()) throw std::runtime_error("Network doesn't have any layers!");
    for (unsigned j = 0; j < numIters; j++) {
        unsigned numLayer = m_layer.size();

        for (int i = numLayer-1; i > 0; i--) {
            StopWatch stopwatch;
            m_layer[i]->backward(
                i==0?input:m_layer[i-1]->getLastOutput(), 
                i==numLayer-1?outputGradients:m_layer[i+1]->getLastInputGradients()
            );
            float time = stopwatch.getElapsedSeconds();
            timePerType[m_layer[i]->layerName()] += time;
        }
    }
    
    float totalTime = 0.0f;
    for (auto &p : timePerType) 
        totalTime += p.second;
    std::cout << "Total backward pass: " << totalTime / numIters * 1000 << " ms" << std::endl;

    for (auto &p : timePerType) {
        std::cout << "Layer type: " << p.first << std::endl;
        std::cout << "  Absolute: " << p.second/numIters*1000 << " ms" << std::endl;
        std::cout << "  Relative: " << p.second/totalTime*100 << " %" << std::endl;
    }
}



void Network::updateParameters(float stepsize)
{
    for (auto &layer : m_layer)
        layer->updateParameters(stepsize);
}


}
