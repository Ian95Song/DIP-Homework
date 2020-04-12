/**
 * Copyright 2019/2020 Andreas Ley
 * Written for Digital Image Processing of TU Berlin
 * For internal use in the course only
 */


#include "Dip6.h"
#include "TaskScheduler.h"
#include "StopWatch.h"
#include "SIMD.h"


#include <typeinfo>
#include <map>
#include <string>
#include <string.h>
#include <stdexcept>
#include <iostream>

#include <immintrin.h>

namespace dip6 {


/**
 * @brief Computes the forward pass of a conv layer
 * @details Don't optimize this one, just get it right.
 * @param input Tensor of input floats (from the previous layer) in layout (height, width, channels, instances)
 * @param kernel Tensor of the filter kernel values in layout (height, width, output channels, input channels)
 * @param bias Tensor of the filter bias values in layout (0, 0, output channels, 0)
 * @param output Tensor of output floats in layout (height, width, channels, instances) in which the result is returned. This gets handed to the next layer.
 */
void reference_convolutionForward(const Tensor &input, const Tensor &kernel, const Tensor &bias, Tensor &output)
{
    // TO DO !!!
    // Loop over all output pixels, output channels, and instances
    for (unsigned y = 0; y < output.getSize(0); y++){
       for (unsigned x = 0; x < output.getSize(1); x++){
          for (unsigned c_o = 0; c_o < output.getSize(2); c_o++){
             for (unsigned instance = 0; instance < input.getSize(3); instance++){
        //     Initialize a summation float to the bias value of the output channel
                float sum = bias(0,0,c_o,0);
             //     Loop over all kernel elements and input channels
                for (unsigned y_k = 0; y_k < kernel.getSize(0); y_k++){
                   for (unsigned x_k = 0; x_k < kernel.getSize(1); x_k++) {
                      for (unsigned c_i = 0; c_i < input.getSize(2); c_i++){
                  //         Add product of kernel and input value to summation float
                         sum += input(y+y_k,x+x_k,c_i,instance)*kernel(y_k,x_k,c_o,c_i);
                      }
                   }
                }
                //     Store summation float to output
                output(y,x,c_o,instance) = sum;
             }
          }
       }
    }

}


/**
 * @brief Computes the backward data pass of a conv layer
 * @details Given the partial error derivatives wrt. to this layer's outputs, computes the partial derivatives wrt. this layer's inputs.
 * @param input Tensor of input floats (from the previous layer) in layout (height, width, channels, instances)
 * @param kernel Tensor of the filter kernel values in layout (height, width, output channels, input channels)
 * @param bias Tensor of the filter bias values in layout (0, 0, output channels, 0)
 * @param output Tensor of output floats in layout (height, width, channels, instances) which was passed to the next layer in the forward pass.
 * @param outputGrad Tensor of gradients in layout (height, width, channels, instances) containing the partial derivatives of the error function wrt. each of this layer's outputs.
 * @param inputGrad Tensor of gradients in layout (height, width, channels, instances) containing the partial derivatives of the error function wrt. each of this layer's inputs.
 */
void reference_convolutionBackwardData(const Tensor &input, const Tensor &kernel, const Tensor &bias, const Tensor &output, const Tensor &outputGrad, Tensor &inputGrad)
{

    if (input.getSize(3) > 64)
        throw std::runtime_error("Not build for more than 64 instances!");

    for (unsigned y = 0; y < input.getSize(0); y++)
        for (unsigned x = 0; x < input.getSize(1); x++)
            for (unsigned c_i = 0; c_i < input.getSize(2); c_i++) {
                float accus[64];
                for (unsigned instance = 0; instance < input.getSize(3); instance++)
                    accus[instance] = 0;

                unsigned minY = std::max<int>(0, (int)y - (int)output.getSize(0) + 1);
                unsigned minX = std::max<int>(0, (int)x - (int)output.getSize(1) + 1);
                unsigned maxY = std::min<int>(y, kernel.getSize(0) - 1);
                unsigned maxX = std::min<int>(x, kernel.getSize(1) - 1);

                for (unsigned y_k = minY; y_k <= maxY; y_k++)
                    for (unsigned x_k = minX; x_k <= maxX; x_k++)
                        for (unsigned c_o = 0; c_o < output.getSize(2); c_o++) {
                            float kernelValue = kernel(y_k, x_k, c_o, c_i);
                            for (unsigned instance = 0; instance < input.getSize(3); instance++) {
                                accus[instance] += outputGrad(y-y_k, x-x_k, c_o, instance) * kernelValue;
                            }
                        }

                for (unsigned instance = 0; instance < input.getSize(3); instance++) {
                    inputGrad(y, x, c_i, instance) = accus[instance];
                }
            }
}

/**
 * @brief Computes the backward parameter pass of a conv layer
 * @details Given the partial error derivatives wrt. to this layer's outputs, computes the partial derivatives wrt. this layer's parameters (weights and bias). Don't optimize this one.
 * @param input Tensor of input floats (from the previous layer) in layout (height, width, channels, instances)
 * @param kernel Tensor of the filter kernel values in layout (height, width, output channels, input channels)
 * @param bias Tensor of the filter bias values in layout (0, 0, output channels, 0)
 * @param output Tensor of output floats in layout (height, width, channels, instances) which was passed to the next layer in the forward pass.
 * @param outputGrad Tensor of gradients in layout (height, width, channels, instances) containing the partial derivatives of the error function wrt. each of this layer's outputs.
 * @param kernelGrad Tensor of gradients in layout (kernel height, kernel width, output channels, input channels) containing the partial derivatives of the error function wrt. each the kernel values.
 * @param biasGrad Tensor of gradients in layout (0, 0, output channels, 0) containing the partial derivatives of the error function wrt. each the bias values.
 */
void reference_convolutionBackwardParameters(const Tensor &input, const Tensor &kernel, const Tensor &bias, Tensor &output, const Tensor &outputGrad, Tensor &kernelGrad, Tensor &biasGrad)
{
    kernelGrad.setZero();
    biasGrad.setZero();
    for (unsigned y = 0; y < output.getSize(0); y++)
        for (unsigned x = 0; x < output.getSize(1); x++)
            for (unsigned c_o = 0; c_o < output.getSize(2); c_o++) {
                for (unsigned y_k = 0; y_k < kernel.getSize(0); y_k++)
                    for (unsigned x_k = 0; x_k < kernel.getSize(1); x_k++)
                        for (unsigned c_i = 0; c_i < input.getSize(2); c_i++)
                            for (unsigned instance = 0; instance < input.getSize(3); instance++)
                                kernelGrad(y_k, x_k, c_o, c_i) += input(y+y_k, x+x_k, c_i, instance) * outputGrad(y, x, c_o, instance);

                for (unsigned instance = 0; instance < input.getSize(3); instance++)
                    biasGrad(0, 0, c_o, 0) += outputGrad(y, x, c_o, instance);
            }
}


/**
 * @brief Computes the forward pass of a conv layer
 * @details Copy and paste reference implementation, parallelize outermost loop, vectorize innermost loop over instances.
 * @param input Tensor of input floats (from the previous layer) in layout (height, width, channels, instances)
 * @param kernel Tensor of the filter kernel values in layout (height, width, output channels, input channels)
 * @param bias Tensor of the filter bias values in layout (0, 0, output channels, 0)
 * @param output Tensor of output floats in layout (height, width, channels, instances) in which the result is returned. This gets handed to the next layer.
 * @param inputChannels Compile-time constant of the number of input channels. Works like a read only variable.
 * @param instances Compile-time constant of the number of instances. Works like a read only variable.
 */
template<unsigned inputChannels,
         unsigned instances>
void convolutionForward_CN(const Tensor &input, const Tensor &kernel, const Tensor &bias, Tensor &output)
{
    // TO DO !!!
    parallelFor(0, output.getSize(0), 1, [&](unsigned y) {
       for (unsigned x = 0; x < output.getSize(1); x++){
          for (unsigned c_o = 0; c_o < output.getSize(2); c_o++){
//             simd::Vector<instances> inp;
             simd::Vector<instances> sum;
//             simd::Scalar kern;
             simd::Scalar bi(&bias(0,0,c_o,0));
             sum = bi;
             for (unsigned y_k = 0; y_k < kernel.getSize(0); y_k++){
                for (unsigned x_k = 0; x_k < kernel.getSize(1); x_k++){
                   for (unsigned c_i = 0; c_i < inputChannels; c_i++){
                      simd::Scalar kern(&kernel(y_k,x_k,c_o,c_i));
                      simd::Vector<instances> inp(&input(y+y_k,x+x_k,c_i,0));
                      sum += inp*kern;
                   }
                }
             }
             sum.store(&output(y,x,c_o,0));
          }
       }


    });


}

/**
 * @brief Computes the backward data pass of a conv layer
 * @details Given the partial error derivatives wrt. to this layer's outputs, computes the partial derivatives wrt. this layer's inputs.
 * Copy and paste reference implementation, parallelize outermost loop, vectorize innermost loop over instances.
 * @param input Tensor of input floats (from the previous layer) in layout (height, width, channels, instances)
 * @param kernel Tensor of the filter kernel values in layout (height, width, output channels, input channels)
 * @param bias Tensor of the filter bias values in layout (0, 0, output channels, 0)
 * @param output Tensor of output floats in layout (height, width, channels, instances) which was passed to the next layer in the forward pass.
 * @param outputGrad Tensor of gradients in layout (height, width, channels, instances) containing the partial derivatives of the error function wrt. each of this layer's outputs.
 * @param inputGrad Tensor of gradients in layout (height, width, channels, instances) containing the partial derivatives of the error function wrt. each of this layer's inputs.
 * @param inputChannels Compile-time constant of the number of input channels. Works like a read only variable.
 * @param outputChannels Compile-time constant of the number of output channels. Works like a read only variable.
 * @param instances Compile-time constant of the number of instances. Works like a read only variable.
 */
template<unsigned inputChannels,
         unsigned outputChannels,
         unsigned instances>
void convolutionBackwardData_ION(const Tensor &input, const Tensor &kernel, const Tensor &bias, const Tensor &output, const Tensor &outputGrad, Tensor &inputGrad)
{
    // TO DO !!!
    if (input.getSize(3) > 64)
    throw std::runtime_error("Not build for more than 64 instances!");

    parallelFor(0, output.getSize(0), 1, [&](unsigned y){
       for (unsigned x = 0; x < output.getSize(1); x++){
          for (unsigned c_i = 0; c_i < input.getSize(2); c_i++){
             simd::Vector<instances>accus(simd::INIT_ZERO);

             unsigned minY = std::max<int>(0, (int)y - (int)output.getSize(0) + 1);
             unsigned minX = std::max<int>(0, (int)x - (int)output.getSize(1) + 1);
             unsigned maxY = std::min<int>(y, kernel.getSize(0) - 1);
             unsigned maxX = std::min<int>(x, kernel.getSize(1) - 1);

             for (unsigned y_k = minY; y_k <= maxY; y_k++){
                for (unsigned x_k = minX; x_k <= maxX; x_k++){
                   for (unsigned c_o = 0; c_o < outputChannels; c_o++){
                      simd::Scalar kernelValue(&kernel(y_k,x_k,c_o,c_i));
                      simd::Vector<instances> outp(&outputGrad(y-y_k,x-x_k,c_o,0));
                      accus += outp*kernelValue;
                   }
                }
             }
             accus.store(&inputGrad(y,x,c_i,0));
          }
       }
    });


}


/**
 * @brief Computes the backward parameter pass of a conv layer
 * @details Given the partial error derivatives wrt. to this layer's outputs, computes the partial derivatives wrt. this layer's parameters (weights and bias). Don't optimize this one.
 * @param input Tensor of input floats (from the previous layer) in layout (height, width, channels, instances)
 * @param kernel Tensor of the filter kernel values in layout (height, width, output channels, input channels)
 * @param bias Tensor of the filter bias values in layout (0, 0, output channels, 0)
 * @param output Tensor of output floats in layout (height, width, channels, instances) which was passed to the next layer in the forward pass.
 * @param outputGrad Tensor of gradients in layout (height, width, channels, instances) containing the partial derivatives of the error function wrt. each of this layer's outputs.
 * @param kernelGrad Tensor of gradients in layout (kernel height, kernel width, output channels, input channels) containing the partial derivatives of the error function wrt. each the kernel values.
 * @param biasGrad Tensor of gradients in layout (0, 0, output channels, 0) containing the partial derivatives of the error function wrt. each the bias values.
 * @param inputChannels Compile-time constant of the number of input channels. Works like a read only variable.
 * @param instances Compile-time constant of the number of instances. Works like a read only variable.
 */
template<unsigned inputChannels,
         unsigned instances>
void convolutionBackwardParameters_CN(const Tensor &input, const Tensor &kernel, const Tensor &bias, Tensor &output, const Tensor &outputGrad, Tensor &kernelGrad, Tensor &biasGrad, std::vector<Tensor> &scratchpad)
{
    enum {
        paddedInputChannels = (inputChannels + 7) & ~7u
    };

    scratchpad.resize(2);

    Tensor &transposedInput = scratchpad[0];
    transposedInput.allocate(
        input.getSize(0),
        input.getSize(1),
        input.getSize(3),
        paddedInputChannels
    );

    parallelFor(0, input.getSize(0), 8, [&](unsigned y) {
        for (unsigned x = 0; x < input.getSize(1); x++) {
            const float *inputPtr = &input(y, x, 0, 0);
            float *transposedInputPtr = &transposedInput(y, x, 0, 0);
#ifndef __SSE__
            for (unsigned c_i = 0; c_i < input.getSize(2); c_i++)
                for (unsigned instance = 0; instance < input.getSize(3); instance++)
                    transposedInputPtr[instance * inputChannels + c_i] = inputPtr[c_i * instances + instance];
#else
            for (unsigned c_i = 0; c_i < (inputChannels & ~3u); c_i+=4)
                for (unsigned instance = 0; instance < instances; instance+=4) {
                    __m128 row0 = _mm_loadu_ps(inputPtr + (c_i + 0) * instances + instance);
                    __m128 row1 = _mm_loadu_ps(inputPtr + (c_i + 1) * instances + instance);
                    __m128 row2 = _mm_loadu_ps(inputPtr + (c_i + 2) * instances + instance);
                    __m128 row3 = _mm_loadu_ps(inputPtr + (c_i + 3) * instances + instance);
                    _MM_TRANSPOSE4_PS(row0, row1, row2, row3);
                    _mm_storeu_ps(transposedInputPtr + (instance + 0) * paddedInputChannels + c_i, row0);
                    _mm_storeu_ps(transposedInputPtr + (instance + 1) * paddedInputChannels + c_i, row1);
                    _mm_storeu_ps(transposedInputPtr + (instance + 2) * paddedInputChannels + c_i, row2);
                    _mm_storeu_ps(transposedInputPtr + (instance + 3) * paddedInputChannels + c_i, row3);
                }
            for (unsigned c_i = inputChannels & ~3u; c_i < inputChannels; c_i++)
                for (unsigned instance = 0; instance < instances; instance++)
                    transposedInputPtr[instance * paddedInputChannels + c_i] = inputPtr[c_i * instances + instance];
            for (unsigned instance = 0; instance < instances; instance++)
                for (unsigned c_i = inputChannels; c_i < paddedInputChannels; c_i++)
                    transposedInputPtr[instance * paddedInputChannels + c_i] = 0.0f;
#endif
        }
    });

    Tensor &paddedKernelGrad = scratchpad[1];
    paddedKernelGrad.allocate(
        kernel.getSize(0),
        kernel.getSize(1),
        kernel.getSize(2),
        paddedInputChannels
    );
    paddedKernelGrad.setZero();


    parallelFor(0, output.getSize(2), 1, [&](unsigned c_o) {
        simd::Vector<simd::SIMD_WIDTH> biasSum(simd::INIT_ZERO);
        for (unsigned y = 0; y < output.getSize(0); y++)
            for (unsigned x = 0; x < output.getSize(1); x++) {
                const float *outputGradPtr = &outputGrad(y, x, c_o, 0);

                for (unsigned y_k = 0; y_k < kernel.getSize(0); y_k++)
                    for (unsigned x_k = 0; x_k < kernel.getSize(1); x_k++) {

                        float *kernelPtr = &paddedKernelGrad(y_k, x_k, c_o, 0);
                        const float *inputPtr = &transposedInput(y+y_k, x+x_k, 0, 0);

                        simd::Vector<paddedInputChannels> accu(kernelPtr);

                        for (unsigned instance = 0; instance < instances; instance++)  {
                            simd::Scalar outputGradValues(outputGradPtr+instance);

                            simd::Vector<paddedInputChannels> inputValue(inputPtr + instance * paddedInputChannels);
                            accu += inputValue * outputGradValues;
                        }

                        accu.store(kernelPtr);
                    }

                for (unsigned instance = 0; instance < instances; instance+=simd::SIMD_WIDTH)
                    biasSum += simd::Vector<simd::SIMD_WIDTH>(outputGradPtr+instance);
            }
        biasSum.storeHorizontalSum(&biasGrad(0, 0, c_o, 0));
    });

    for (unsigned y_k = 0; y_k < kernel.getSize(0); y_k++)
        for (unsigned x_k = 0; x_k < kernel.getSize(1); x_k++)
            for (unsigned c_o = 0; c_o < kernel.getSize(2); c_o++)
                for (unsigned c_i = 0; c_i < inputChannels; c_i++)
                    kernelGrad(y_k, x_k, c_o, c_i) = paddedKernelGrad(y_k, x_k, c_o, c_i);

}


void convolutionForward(const Tensor &input, const Tensor &kernel, const Tensor &bias, Tensor &output);
void convolutionBackwardData(const Tensor &input, const Tensor &kernel, const Tensor &bias, const Tensor &output, const Tensor &outputGrad, Tensor &inputGrad);
void convolutionBackwardParameters(const Tensor &input, const Tensor &kernel, const Tensor &bias, Tensor &output, const Tensor &outputGrad, Tensor &kernelGrad, Tensor &biasGrad, std::vector<Tensor> &scratchpad);


namespace layers {


/**
 * @brief Computes the forward pass of a ReLU layer
 * @param input Tensor of input floats (from the previous layer) in layout (height, width, channels, instances)
 * @details Vectorize with 8-wide vectors, don't use multithreading.
 * Write the output to m_output. It is a dip6::Tensor of the same shape as input.
 */
void ReLU::forward(const Tensor &input)
{
    m_output.allocateLike(input);
    simd::Vector<8> inp(simd::INIT_ZERO);
    simd::Vector<8> outp(simd::INIT_ZERO);
    simd::Vector<8> null(simd::INIT_ZERO);
    for (unsigned i = 0; i < input.getTotalSize(); i+=8){
       inp.load(&input[i]);
       outp = simd::max(inp,null);
       outp.store(&m_output[i]);
    }

    // TO DO !!!

/*
    Remember that:
    for (unsigned i = 0; i < someTensor.getSize(0); i++)
        for (unsigned j = 0; j < someTensor.getSize(1); j++)
            for (unsigned k = 0; k < someTensor.getSize(2); k++)
                for (unsigned l = 0; l < someTensor.getSize(3); l++)
                    someTensor(i, j, k, l) = ....

    is equivalent to:
    for (unsigned i = 0; i < someTensor.getTotalSize(); i++)
        someTensor[i] = ...


    === simd::Vector cheat sheet ===
    Not just for this function but also for the other ones.


    simd::Vector<8> a, b, c; // 8 element vector of uninitialized values
    a.setZero(); // set all to zero
    a.load(&someTensor(a, b, c, d)); // loads values someTensor(a, b, c, d+0) ... someTensor(a, b, c, d+7) into a
    a.load(&someTensor[a]); // loads values someTensor[a+0] ... someTensor[a+7] into a

    a.store(&someTensor(a, b, c, d)); // writes values from a to someTensor(a, b, c, d+0) ... someTensor(a, b, c, d+7)
    a.store(&someTensor[a]); // writes values from a to someTensor[a+0] ... someTensor[a+7]

    simd::Vector<8> a(simd::INIT_ZERO); // shorthand for instantiation + initialization with zero
    simd::Vector<8> a(&someTensor[a]); // shorthand for instantiation + load

    All basic operations are pair wise:
    c = a+b;
    c = a-b;
    c *= a/b;
    ...

    c = simd::max(a, b) // Pair wise chooses the maximum of a and b
    c = simd::min(a, b) // Pair wise chooses the minimum of a and b

    simd::Vector<8> mask = a > b; // Pair wise comparison. The outcome is stored in mask. All of these are implemented: <, >, <=, >=, ==, !=
    c = simd::select(mask, a, b); Selects element wise between a and b (in this setup equivalent to simd::max(a, b))

    simd::Vector<8> mask = a > b; // Pair wise comparison. The outcome is stored in mask
    c = simd::selectOrZero(mask, a); Selects element wise between a and zero


    Sometimes a single float needs to be used on all elements of a vector (e.g. set all elements to the bias or multiply everything by one kernel factor):
    simd::Vector<8> a, b;
    simd::Scalar f;
    f.load(&someTensor(i, j, k, l));  // loads just one value into f
    f.setZero()  // set f to zero

    a = f; // Set all elements in a to the one element in f
    a = b + f; // Add the one element in f to each element in b and store in a

    simd::Scalar f(simd::INIT_ZERO); // shorthand for instantiation + initialization with zero
    simd::Scalar f(&someTensor(i, j, k, l)); // shorthand for instantiation + load
*/

}

/**
 * @brief Computes the backward data pass of a ReLU layer
 * @param input Tensor of input floats (from the previous layer) in layout (height, width, channels, instances)
 * @param outputGradients Tensor of gradients in layout (height, width, channels, instances) containing the partial derivatives of the error function wrt. each of this layer's outputs.
 * @details Given the partial error derivatives wrt. to this layer's outputs, computes the partial derivatives wrt. this layer's inputs.
 * Vectorize with 8-wide vectors, don't use multithreading.
 * Write the result to m_inputGradients. It is a dip6::Tensor of the same shape as input.
 */
void ReLU::backward(const Tensor &input, const Tensor &outputGradients)
{
    m_inputGradients.allocateLike(input);
    // TO DO !!!
    simd::Vector<8> inp(simd::INIT_ZERO);
    simd::Vector<8> outpg(simd::INIT_ZERO);
    simd::Vector<8> null(simd::INIT_ZERO);
    for (unsigned i = 0; i < input.getTotalSize(); i+=8){
       inp.load(&input[i]);
       outpg.load(&outputGradients[i]);
       simd::Vector<8> mask = inp > null;
       simd::Vector<8> vorz = simd::selectOrZero(mask,outpg);
       vorz.store(&m_inputGradients[i]);
    }


}
}



namespace optimizer {

void MomentumSGD::performStep(float stepsize)
{
    const float oneMinusMomentum = 1.0f - m_momentum;
    for (unsigned i = 0; i < m_parameterMomentum.size(); i++) {
        Tensor &Tmomentum = m_parameterMomentum[i];
        Tensor &Tgradients = m_layer->getParameterGradients()[i];
        Tensor &Tvalues = m_layer->getParameters()[i];

        for (unsigned i = 0; i < Tvalues.getTotalSize(); i++) {
            float g = Tgradients[i];


            // I forgot to mention this part in the tutorial:
            // It is sometimes beneficial to penalize large parameters (weights).
            // Here, we add a small fraction of the value to the gradient. This
            // is equivalent to an additional loss term that incorporates the
            // sum over all squared parameters.
            g += Tvalues[i] * m_L2Loss;
            // Just copy the line for Adam, or, if it confuses you, leave it out.
            // The difference is marginal.


            float m = Tmomentum[i];
            m = m * m_momentum + g * oneMinusMomentum;

            Tvalues[i] -= m * stepsize;

            Tmomentum[i] = m;
        }
    }
}

/**
 * @brief Updates parameters according to the Adam scheme
 * @param stepsize The learning rate
 */
void Adam::performStep(float stepsize)
{
    const float oneMinusBeta1 = 1.0f - m_beta1;
    const float oneMinusBeta2 = 1.0f - m_beta2;
    for (unsigned i = 0; i < m_parameterMomentum.size(); i++) {
        Tensor &Tvariance = m_parameterGradExpectation[i];
        Tensor &Tmomentum = m_parameterMomentum[i];
        Tensor &Tgradients = m_layer->getParameterGradients()[i];
        Tensor &Tvalues = m_layer->getParameters()[i];

        for (unsigned i = 0; i < Tvalues.getTotalSize(); i++) {
            // TO DO !!!

            // This loops over all parameters so that each element can be handled individually.

            // Tvalues[i] is the current value of the parameter. This is supposed to be updated.

            // Tgradients[i] is the computed gradient, i.e. the result of the forward+backward passes.

            // Tmomentum[i] is the momentum or running average of the gradient. Use this read the last
            // iteration's momentum and store the updated momentum for the next iteration.
            // Use m_beta1 as the \beta_1 parameter in the formula.

            // Tvariance[i] is the estimated variance or running average of the squared gradient. Use this read the last
            // iteration's variance and store the updated variance for the next iteration. (Yes it works almost like the momentum thing)
            // Use m_beta2 as the \beta_2 parameter in the formula.

            // As the stabilization parameter \epsilon use 1e-8f
           float g = Tgradients[i];


            // I forgot to mention this part in the tutorial:
            // It is sometimes beneficial to penalize large parameters (weights).
            // Here, we add a small fraction of the value to the gradient. This
            // is equivalent to an additional loss term that incorporates the
            // sum over all squared parameters.
           g += Tvalues[i] * m_L2Loss;
           float ep = 1e-8f;
           Tmomentum[i] = m_beta1 * Tmomentum[i] + oneMinusBeta1 * g;
           Tvariance[i] = m_beta2 * Tvariance[i] + oneMinusBeta2 * g * g;
           Tvalues[i] = Tvalues[i] - (stepsize * Tmomentum[i] / (std::sqrt(Tvariance[i])+ep));
        }
    }
}

}


/**
 * @brief Compute the Mean Squared Error loss as well as the gradients to feed back into the network.
 * @param computedOutput The output that the network did produce
 * @param desiredOutput The output that the network was supposed to produce
 * @param computedOutputGradients Tensor to return in the computed partial derivate of the loss wrt. each element in the computedOutput tensor
 * @return The computed loss
 */
float MSELoss::computeLoss(const Tensor &computedOutput, const Tensor &desiredOutput, Tensor &computedOutputGradients)
{
    computedOutputGradients.allocateLike(computedOutput);

    unsigned shiftX = (desiredOutput.getSize(1) - computedOutput.getSize(1))/2;
    unsigned shiftY = (desiredOutput.getSize(0) - computedOutput.getSize(0))/2;
    float err, sum = 0.;

    for (unsigned y = 0; y < computedOutput.getSize(0); y++){
       for (unsigned x = 0; x < computedOutput.getSize(1); x++){
          for (unsigned c_o = 0; c_o < computedOutput.getSize(2); c_o++){
             for (unsigned instance = 0; instance < computedOutput.getSize(3); instance++){
                err = computedOutput(y,x,c_o,instance)-desiredOutput(y+shiftY,x+shiftX,c_o,instance);
                computedOutputGradients(y,x,c_o,instance) = 2.0f * err / computedOutput.getTotalSize();
                sum += pow(err,2);
             }
          }
       }
    }
    float l = sum/computedOutput.getTotalSize();
    // TO DO !!!



    // No SIMD or multithreading needed in here.

    // Compute the loss and the gradients!
    // The loss (scalar value, aberage over all pixels, channels, and instances) is returned as the function's return value.
    // The derivatives of the loss wrt. each of the network's outputs, i.e., \frac{\partial E}{\partial y}, is returned in
    // computedOutputGradients which has the same shape as computedOutput.

    // Keep in mind that computedOutput is smaller than desiredOutput and that only the centered, inner region of desiredOutput
    // which is covered by computedOutput should be considered.
    // In line with this, when dividing by the number of element (to compute the average from the sum), use the number of elements
    // in computedOutput, and not the number of elements in desiredOutput.

    // The loss is:
    //        E = \frac{1}{h \cdot w \cdot c \cdot n} \cdot \sum_{h, w, c, n} (y_{h,w,c,n} - \hat{y}_{h+s_y,w+s_x,c,n})^2
    // Then its derivative is:
    //        \frac{\partial E}{\partial y_{h, w, c, n}} = \frac{2}{h \cdot w \cdot c \cdot n} \cdot \sum_{h, w, c, n} (y_{h,w,c,n} - \hat{y}_{h+s_y,w+s_x,c,n})

    return l;
}




/////////////////////////////////////////////////////////
/////                 Given stuff                   /////
/////////////////////////////////////////////////////////


namespace layers {

ConvReference::ConvReference(unsigned kernelWidth, unsigned kernelHeight, unsigned inputChannels, unsigned outputChannels) : Conv(kernelWidth, kernelHeight, inputChannels, outputChannels)
{
}

void ConvReference::forward(const Tensor &input)
{
    Conv::resizeOutput(input);

    reference_convolutionForward(input, m_parameterTensors[PARAM_KERNEL], m_parameterTensors[PARAM_BIAS], m_output);
}

void ConvReference::backward(const Tensor &input, const Tensor &outputGradients)
{
    Conv::resizeInputGradients(input, outputGradients);

    reference_convolutionBackwardParameters(
        input,
        m_parameterTensors[PARAM_KERNEL],
        m_parameterTensors[PARAM_BIAS],
        m_output,
        outputGradients,
        m_parameterGradientTensors[PARAM_KERNEL],
        m_parameterGradientTensors[PARAM_BIAS]
    );
    reference_convolutionBackwardData(
        input,
        m_parameterTensors[PARAM_KERNEL],
        m_parameterTensors[PARAM_BIAS],
        m_output,
        outputGradients,
        m_inputGradients
    );
}



ConvOptimized::ConvOptimized(unsigned kernelWidth, unsigned kernelHeight, unsigned inputChannels, unsigned outputChannels) : Conv(kernelWidth, kernelHeight, inputChannels, outputChannels)
{
}


void ConvOptimized::forward(const Tensor &input)
{
    Conv::resizeOutput(input);

    convolutionForward(input, m_parameterTensors[PARAM_KERNEL], m_parameterTensors[PARAM_BIAS], m_output);
}

void ConvOptimized::backward(const Tensor &input, const Tensor &outputGradients)
{
    Conv::resizeInputGradients(input, outputGradients);

    convolutionBackwardParameters(
        input,
        m_parameterTensors[PARAM_KERNEL],
        m_parameterTensors[PARAM_BIAS],
        m_output,
        outputGradients,
        m_parameterGradientTensors[PARAM_KERNEL],
        m_parameterGradientTensors[PARAM_BIAS],
        m_scratchpad
    );

    convolutionBackwardData(
        input,
        m_parameterTensors[PARAM_KERNEL],
        m_parameterTensors[PARAM_BIAS],
        m_output,
        outputGradients,
        m_inputGradients
    );
}

}


namespace optimizer {

MomentumSGD::MomentumSGD(ParametrizedLayer *layer, float momentum, float L2Loss) : Optimizer(layer), m_momentum(momentum), m_L2Loss(L2Loss)
{
    m_parameterMomentum.resize(m_layer->getParameters().size());
    for (unsigned i = 0; i < m_parameterMomentum.size(); i++) {
        m_parameterMomentum[i].allocateLike(m_layer->getParameters()[i]);
        m_parameterMomentum[i].setZero();
    }
}

void MomentumSGD::saveSnapshot(std::ostream &stream)
{
    for (const auto &t : m_parameterMomentum)
        stream.write((const char*)&t(0, 0, 0, 0), t.getTotalSize() * sizeof(float));
}

void MomentumSGD::restoreSnapshot(std::istream &stream)
{
    for (auto &t : m_parameterMomentum)
        stream.read((char*)&t(0, 0, 0, 0), t.getTotalSize() * sizeof(float));
}


Adam::Adam(ParametrizedLayer *layer, float L2Loss, float beta1, float beta2) : Optimizer(layer), m_L2Loss(L2Loss), m_beta1(beta1), m_beta2(beta2)
{
    m_parameterMomentum.resize(m_layer->getParameters().size());
    for (unsigned i = 0; i < m_parameterMomentum.size(); i++) {
        m_parameterMomentum[i].allocateLike(m_layer->getParameters()[i]);
        m_parameterMomentum[i].setZero();
    }
    m_parameterGradExpectation.resize(m_layer->getParameters().size());
    for (unsigned i = 0; i < m_parameterGradExpectation.size(); i++) {
        m_parameterGradExpectation[i].allocateLike(m_layer->getParameters()[i]);
        m_parameterGradExpectation[i].setZero();
    }
}


void Adam::saveSnapshot(std::ostream &stream)
{
    for (const auto &t : m_parameterMomentum)
        stream.write((const char*)&t(0, 0, 0, 0), t.getTotalSize() * sizeof(float));
    for (const auto &t : m_parameterGradExpectation)
        stream.write((const char*)&t(0, 0, 0, 0), t.getTotalSize() * sizeof(float));
}

void Adam::restoreSnapshot(std::istream &stream)
{
    for (auto &t : m_parameterMomentum)
        stream.read((char*)&t(0, 0, 0, 0), t.getTotalSize() * sizeof(float));
    for (auto &t : m_parameterGradExpectation)
        stream.read((char*)&t(0, 0, 0, 0), t.getTotalSize() * sizeof(float));
}

}







template<unsigned instances>
void convolutionForward_N(const Tensor &input, const Tensor &kernel, const Tensor &bias, Tensor &output)
{
    switch (input.getSize(2)) {
#define DISPATCH(n) \
        case (n): convolutionForward_CN<(n), instances>(input, kernel, bias, output); break;

//        DISPATCH(1)
//        DISPATCH(2)
        DISPATCH(3)
//        DISPATCH(4)
//        DISPATCH(8)
        DISPATCH(16)
        DISPATCH(32)
        DISPATCH(64)
//        DISPATCH(128)
        default:
            throw std::runtime_error("No forward convolution implementation for the desired number of input channels was compiled!");
    }
#undef DISPATCH
}


void convolutionForward(const Tensor &input, const Tensor &kernel, const Tensor &bias, Tensor &output)
{
    switch (input.getSize(3)) {
#define DISPATCH(n) \
        case (n): convolutionForward_N<(n)>(input, kernel, bias, output); break;

        DISPATCH(8)
        DISPATCH(16)
        DISPATCH(32)
//        DISPATCH(64)
//        DISPATCH(128)
        default:
            throw std::runtime_error("No forward convolution implementation for the desired batch size was compiled!");
    }
#undef DISPATCH
}



template<unsigned outputChannels,
         unsigned instances>
void convolutionBackwardData_ON(const Tensor &input, const Tensor &kernel, const Tensor &bias, const Tensor &output, const Tensor &outputGrad, Tensor &inputGrad)
{
    switch (input.getSize(2)) {
#define DISPATCH(n) \
        case (n): convolutionBackwardData_ION<(n), outputChannels, instances>(input, kernel, bias, output, outputGrad, inputGrad); break;

//        DISPATCH(1)
//        DISPATCH(2)
        DISPATCH(3)
//        DISPATCH(4)
//        DISPATCH(8)
        DISPATCH(16)
        DISPATCH(32)
        DISPATCH(64)
//        DISPATCH(128)
        default:
            throw std::runtime_error("No backward data convolution implementation for the desired number of input channels was compiled!");
    }
#undef DISPATCH
}

template<unsigned instances>
void convolutionBackwardData_N(const Tensor &input, const Tensor &kernel, const Tensor &bias, const Tensor &output, const Tensor &outputGrad, Tensor &inputGrad)
{
    switch (output.getSize(2)) {
#define DISPATCH(n) \
        case (n): convolutionBackwardData_ON<(n), instances>(input, kernel, bias, output, outputGrad, inputGrad); break;

//        DISPATCH(1)
//        DISPATCH(2)
        DISPATCH(3)
//        DISPATCH(4)
//        DISPATCH(8)
        DISPATCH(16)
        DISPATCH(32)
        DISPATCH(64)
//        DISPATCH(128)
        default:
            throw std::runtime_error("No backward data convolution implementation for the desired number of output channels was compiled!");
    }
#undef DISPATCH
}

void convolutionBackwardData(const Tensor &input, const Tensor &kernel, const Tensor &bias, const Tensor &output, const Tensor &outputGrad, Tensor &inputGrad)
{
    switch (input.getSize(3)) {
#define DISPATCH(n) \
        case (n): convolutionBackwardData_N<(n)>(input, kernel, bias, output, outputGrad, inputGrad); break;

        DISPATCH(8)
        DISPATCH(16)
        DISPATCH(32)
//        DISPATCH(64)
//        DISPATCH(128)
        default:
            throw std::runtime_error("No backward data convolution implementation for the desired batch size was compiled!");
    }
#undef DISPATCH
}


template<unsigned instances>
void convolutionBackwardParameters_N(const Tensor &input, const Tensor &kernel, const Tensor &bias, Tensor &output, const Tensor &outputGrad, Tensor &kernelGrad, Tensor &biasGrad, std::vector<Tensor> &scratchpad)
{
    switch (input.getSize(2)) {
#define DISPATCH(n) \
        case (n): convolutionBackwardParameters_CN<(n), instances>(input, kernel, bias, output, outputGrad, kernelGrad, biasGrad, scratchpad); break;

//        DISPATCH(1)
//        DISPATCH(2)
        DISPATCH(3)
//        DISPATCH(4)
//        DISPATCH(8)
        DISPATCH(16)
        DISPATCH(32)
        DISPATCH(64)
//        DISPATCH(128)
        default:
            throw std::runtime_error("No backward parameter convolution implementation for the desired number of input channels was compiled!");
    }
#undef DISPATCH
}


void convolutionBackwardParameters(const Tensor &input, const Tensor &kernel, const Tensor &bias, Tensor &output, const Tensor &outputGrad, Tensor &kernelGrad, Tensor &biasGrad, std::vector<Tensor> &scratchpad)
{
    switch (input.getSize(3)) {
#define DISPATCH(n) \
        case (n): convolutionBackwardParameters_N<(n)>(input, kernel, bias, output, outputGrad, kernelGrad, biasGrad, scratchpad); break;

        DISPATCH(8)
        DISPATCH(16)
        DISPATCH(32)
//        DISPATCH(64)
//        DISPATCH(128)
        default:
            throw std::runtime_error("No backward parameter convolution implementation for the desired batch size was compiled!");
    }
#undef DISPATCH
}

}
