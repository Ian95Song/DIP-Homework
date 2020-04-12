/**
 * Copyright 2019/2020 Andreas Ley
 * Written for Digital Image Processing of TU Berlin
 * For internal use in the course only
 */


#ifndef NETWORK_H
#define NETWORK_H

#include "Tensor.h"

#include <vector>
#include <memory>
#include <istream>
#include <random>
#include <ostream>

namespace dip6 {

class ParametrizedLayer;

class Optimizer
{
    public:
        Optimizer(ParametrizedLayer *layer) : m_layer(layer) { }
        virtual ~Optimizer() = default;
        virtual void performStep(float stepsize) = 0;

        virtual void saveSnapshot(std::ostream &stream) {}
        virtual void restoreSnapshot(std::istream &stream) {}
    protected:
        ParametrizedLayer *m_layer;
};

class Layer
{
    public:
        virtual ~Layer() = default;
        
        virtual const char *layerName() = 0;
        
        virtual void forward(const Tensor &input) = 0;
        virtual void backward(const Tensor &input, const Tensor &outputGradients) = 0;
        inline const Tensor &getLastOutput() const { return m_output; }
        inline const Tensor &getLastInputGradients() const { return m_inputGradients; }

        virtual void updateParameters(float stepsize) { }
        virtual void saveSnapshot(std::ostream &stream) { }
        virtual void restoreSnapshot(std::istream &stream) { }
    protected:
        Tensor m_output;
        Tensor m_inputGradients;
};


class ParametrizedLayer : public Layer
{
    public:
        virtual ~ParametrizedLayer() = default;

        virtual void updateParameters(float stepsize) override;
        
        inline std::vector<Tensor> &getParameters() { return m_parameterTensors; }
        inline std::vector<Tensor> &getParameterGradients() { return m_parameterGradientTensors; }
        inline Optimizer *getOptimizer() { return m_optimizer.get(); }
        
        virtual void saveSnapshot(std::ostream &stream) override;
        virtual void restoreSnapshot(std::istream &stream) override;
    protected:
        std::vector<Tensor> m_parameterTensors;
        std::vector<Tensor> m_parameterGradientTensors;
        std::unique_ptr<Optimizer> m_optimizer;
        
        void allocateGradientTensors();
};

namespace layers {
    
class Conv : public ParametrizedLayer
{
    public:
        Conv(unsigned kernelWidth, unsigned kernelHeight, unsigned inputChannels, unsigned outputChannels);
        
        virtual const char *layerName() override { return "Conv"; }
        
        template<class Optimizer, typename... Args>
        Conv &setOptimizer(Args&&... args) { 
            m_optimizer.reset(new Optimizer(this, std::forward<Args>(args)...)); 
            return *this; 
        }
        
        Conv &initialize(std::mt19937 &rng, float kernelScale = 1.0f, float biasScale = 1e-2f);

        enum {
            PARAM_KERNEL,
            PARAM_BIAS,
            NUM_PARAMS
        };
    protected:
        void resizeOutput(const Tensor &input);
        void resizeInputGradients(const Tensor &input, const Tensor &outputGradients);
};

class Upsample : public Layer
{
    public:
        Upsample(unsigned upsampleX, unsigned upsampleY);
        
        virtual const char *layerName() override { return "Upsample"; }

        virtual void forward(const Tensor &input) override;
        virtual void backward(const Tensor &input, const Tensor &outputGradients) override;
    protected:
        unsigned m_upsampleX;
        unsigned m_upsampleY;
};

class AvgPool : public Layer
{
    public:
        AvgPool(unsigned downsampleX, unsigned downsampleY);
        
        virtual const char *layerName() override { return "AvgPool"; }

        virtual void forward(const Tensor &input) override;
        virtual void backward(const Tensor &input, const Tensor &outputGradients) override;
    protected:
        unsigned m_downsampleX;
        unsigned m_downsampleY;
};


}

class Network
{
    public:
        template<class LayerType, typename... Args>
        LayerType &appendLayer(Args&&... args) { 
            LayerType *p; 
            m_layer.push_back(std::unique_ptr<Layer>(p = new LayerType(std::forward<Args>(args)...))); 
            return *p; 
        }

        const Tensor &forward(const Tensor &input);
        void backward(const Tensor &input, const Tensor &outputGradients);
        void updateParameters(float stepsize);
        

        void saveSnapshot(std::ostream &stream);
        void restoreSnapshot(std::istream &stream);

        void saveSnapshot(const std::string &filename);
        void restoreSnapshot(const std::string &filename);
        
        void benchmarkForward(const Tensor &input);
        void benchmarkBackward(const Tensor &input, const Tensor &outputGradients);
    protected:
        std::vector<std::unique_ptr<Layer>> m_layer;
};


class DataProvider 
{
    public:
        virtual ~DataProvider() = default;
        
        virtual void reset() = 0;
        virtual bool fetchMinibatch(Tensor &input, Tensor &desiredOutput) = 0;
    protected:
};


class Loss
{
    public:
        virtual ~Loss() = default;
        
        virtual float computeLoss(const Tensor &computedOutput, const Tensor &desiredOutput, Tensor &computedOutputGradients) = 0;
    protected:
};

}

#endif // NETWORK_H
