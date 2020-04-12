/**
 * Copyright 2019/2020 Andreas Ley
 * Written for Digital Image Processing of TU Berlin
 * For internal use in the course only
 */

#include "Tensor.h"

namespace dip6 {
        
void Tensor::allocate(unsigned size0, unsigned size1, unsigned size2, unsigned size3)
{
    if ((m_size[0] == size0) && (m_size[1] == size1) && (m_size[2] == size2) && (m_size[3] == size3))
        return;
    
    m_size[0] = size0;
    m_size[1] = size1;
    m_size[2] = size2;
    m_size[3] = size3;
    
    m_stride[3] = 1;
    m_stride[2] = m_stride[3] * m_size[3];
    m_stride[1] = m_stride[2] * m_size[2];
    m_stride[0] = m_stride[1] * m_size[1];
    m_totalSize = m_stride[0] * m_size[0];
    
    allocateAligned(m_totalSize);
}

void Tensor::allocateLike(const Tensor &other)
{
    allocate(other.getSize(0), other.getSize(1), other.getSize(2), other.getSize(3));
}


void Tensor::allocateAligned(unsigned numFloats)
{
    m_data.resize(numFloats*sizeof(float) + TENSOR_ALIGNMENT_BYTES);
    std::size_t ptr = (std::size_t)m_data.data();
    
    ptr = (ptr + TENSOR_ALIGNMENT_BYTES-1) & ~(TENSOR_ALIGNMENT_BYTES-1);
    
    m_alignedDataPtr = (float*)ptr;
}

        
}
