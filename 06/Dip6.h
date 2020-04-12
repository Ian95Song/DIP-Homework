/**
 * Copyright 2019/2020 Andreas Ley
 * Written for Digital Image Processing of TU Berlin
 * For internal use in the course only
 */


#ifndef DIP_H_
#define DIP_H_

#include "Tensor.h"
#include "Network.h"


namespace dip6 {
    
namespace layers {

class ConvReference : public Conv
{
    public:
        ConvReference(unsigned kernelWidth, unsigned kernelHeight, unsigned inputChannels, unsigned outputChannels);
        
        virtual const char *layerName() override { return "ConvReference"; }
        virtual void forward(const Tensor &input) override;
        virtual void backward(const Tensor &input, const Tensor &outputGradients) override;
};

class ConvOptimized : public Conv
{
    public:
        ConvOptimized(unsigned kernelWidth, unsigned kernelHeight, unsigned inputChannels, unsigned outputChannels);
        
        virtual const char *layerName() override { return "ConvOptimized"; }
        
        virtual void forward(const Tensor &input) override;
        virtual void backward(const Tensor &input, const Tensor &outputGradients) override;
    protected:
        std::vector<Tensor> m_scratchpad;
};


class ReLU : public Layer
{
    public:
        virtual const char *layerName() override { return "ReLU"; }
        
        virtual void forward(const Tensor &input) override;
        virtual void backward(const Tensor &input, const Tensor &outputGradients) override;
};

}

namespace optimizer {

class MomentumSGD : public Optimizer
{
    public:
        MomentumSGD(ParametrizedLayer *layer, float L2Loss = 1e-5f, float momentum = 0.2f);
        virtual ~MomentumSGD() = default;
        virtual void performStep(float stepsize) override;

        virtual void saveSnapshot(std::ostream &stream) override;
        virtual void restoreSnapshot(std::istream &stream) override;
    protected:
        std::vector<Tensor> m_parameterMomentum;
        float m_momentum;
        float m_L2Loss;
};

class Adam : public Optimizer
{
    public:
        Adam(ParametrizedLayer *layer, float L2Loss = 1e-5f, float beta1 = 0.9f, float beta2 = 0.999f);
        virtual ~Adam() = default;
        virtual void performStep(float stepsize) override;

        virtual void saveSnapshot(std::ostream &stream) override;
        virtual void restoreSnapshot(std::istream &stream) override;
    protected:
        std::vector<Tensor> m_parameterMomentum;
        std::vector<Tensor> m_parameterGradExpectation;
        float m_beta1, m_beta2;
        float m_L2Loss;
};

}

class MSELoss : public Loss
{
    public:
        virtual float computeLoss(const Tensor &computedOutput, const Tensor &desiredOutput, Tensor &computedOutputGradients) override;
};

}

#endif
