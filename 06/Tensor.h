/**
 * Copyright 2019/2020 Andreas Ley
 * Written for Digital Image Processing of TU Berlin
 * For internal use in the course only
 */

#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <string.h>
#include <stdexcept>

namespace dip6 {
    
enum {
    TENSOR_ALIGNMENT_FLOATS = 8,
    TENSOR_ALIGNMENT_BYTES = TENSOR_ALIGNMENT_FLOATS * sizeof(float),
};
 
    
class Tensor
{
    public:
        // These are deleted on purpose so you don't try to waste processing power copying tensors around
        Tensor(const Tensor&&) = delete;
        void operator=(const Tensor&&) = delete;
        Tensor(const Tensor& tensor) {
            if ((tensor.m_alignedDataPtr != nullptr) || (m_alignedDataPtr != nullptr))
                throw std::runtime_error("Don't copy tensors");
        }
        void operator=(const Tensor& tensor) {
            if ((tensor.m_alignedDataPtr != nullptr) || (m_alignedDataPtr != nullptr))
                throw std::runtime_error("Don't copy tensors");
        }



        Tensor() = default;

        void allocate(unsigned size0, unsigned size1, unsigned size2, unsigned size3);
        void allocateLike(const Tensor &other);
        
        inline unsigned getSize(unsigned dim) const { return m_size[dim]; }
        inline unsigned getTotalSize() const { return m_totalSize; }
        inline float &operator[](unsigned i) { return m_alignedDataPtr[i]; }
        inline const float &operator[](unsigned i) const { return m_alignedDataPtr[i]; }
        
        inline float &operator()(unsigned i, unsigned j, unsigned k, unsigned l) {
            return m_alignedDataPtr[
                    i * m_stride[0] +
                    j * m_stride[1] +
                    k * m_stride[2] +
                    l * m_stride[3]
            ];
        }
        inline const float &operator()(unsigned i, unsigned j, unsigned k, unsigned l) const {
            return m_alignedDataPtr[
                    i * m_stride[0] +
                    j * m_stride[1] +
                    k * m_stride[2] +
                    l * m_stride[3]
            ];
        }
        
        inline void setZero() { memset(m_alignedDataPtr, 0x00, m_totalSize * sizeof(float)); }
    protected:
        unsigned m_size[4] = {};
        unsigned m_stride[4] = {};
        unsigned m_totalSize = 0;
        
        void allocateAligned(unsigned numFloats);
        
        std::vector<unsigned char> m_data;
        float *m_alignedDataPtr = nullptr;
};

}

#endif // TENSOR_H
