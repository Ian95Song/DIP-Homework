/**
 * Copyright 2019/2020 Andreas Ley
 * Written for Digital Image Processing of TU Berlin
 * For internal use in the course only
 */

#ifndef SIMD_H
#define SIMD_H

#include <algorithm>
#include <cstdint>
#include <immintrin.h>

#define DIP6_USE_AVX
#define DIP6_USE_SSE

namespace dip6 {
namespace simd {

    
enum InitZeroEnum {
    INIT_ZERO
};
    
#if defined(__AVX__) && defined(DIP6_USE_AVX)

enum {
    SIMD_WIDTH = 8,
};
    
struct Scalar
{
    Scalar() = default;
    Scalar(const float *src) { load(src); }
    inline void load(const float *src) { v = _mm256_broadcast_ss(src); }
    Scalar(InitZeroEnum) { setZero(); }
    inline void setZero() { v = _mm256_setzero_ps(); }    
    __m256 v;
};

template<unsigned width>
struct Vector 
{
    enum {
        NUM_LANES = width,
        NUM_REGS = width / SIMD_WIDTH
    };
    
    Vector() = default;
    Vector(const float *src) { load(src); }
    Vector(const Scalar &v) { (*this) = v; }
    Vector(InitZeroEnum) { setZero(); }
    
    inline void load(const float *src) {
        for (unsigned i = 0; i < NUM_REGS; i++)
            v[i] = _mm256_load_ps(src + i*SIMD_WIDTH);
    }
    inline void store(float *dst) const {
        for (unsigned i = 0; i < NUM_REGS; i++)
            _mm256_store_ps(dst + i*SIMD_WIDTH, v[i]);
    }
    
    inline void setZero() {
        for (unsigned i = 0; i < NUM_REGS; i++)
            v[i] = _mm256_setzero_ps();
    }
    
    inline void operator=(const Vector<width> &other) {
        for (unsigned i = 0; i < NUM_REGS; i++)
            v[i] = other.v[i];
    }
    inline void operator=(const Scalar &other) {
        for (unsigned i = 0; i < NUM_REGS; i++)
            v[i] = other.v;
    }
    
    inline Vector<width> operator+(const Vector<width> &other) {
        Vector<width> result;
        for (unsigned i = 0; i < NUM_REGS; i++)
            result.v[i] = _mm256_add_ps(v[i], other.v[i]);
        return result;
    }
    inline Vector<width> operator-(const Vector<width> &other) {
        Vector<width> result;
        for (unsigned i = 0; i < NUM_REGS; i++)
            result.v[i] = _mm256_sub_ps(v[i], other.v[i]);
        return result;
    }
    inline Vector<width> operator*(const Vector<width> &other) {
        Vector<width> result;
        for (unsigned i = 0; i < NUM_REGS; i++)
            result.v[i] = _mm256_mul_ps(v[i], other.v[i]);
        return result;
    }
    inline Vector<width> operator/(const Vector<width> &other) {
        Vector<width> result;
        for (unsigned i = 0; i < NUM_REGS; i++)
            result.v[i] = _mm256_div_ps(v[i], other.v[i]);
        return result;
    }
    
    inline void operator+=(const Vector<width> &other) {
        for (unsigned i = 0; i < NUM_REGS; i++)
            v[i] = _mm256_add_ps(v[i], other.v[i]);
    }
    inline void operator-=(const Vector<width> &other) {
        for (unsigned i = 0; i < NUM_REGS; i++)
            v[i] = _mm256_sub_ps(v[i], other.v[i]);
    }
    inline void operator*=(const Vector<width> &other) {
        for (unsigned i = 0; i < NUM_REGS; i++)
            v[i] = _mm256_mul_ps(v[i], other.v[i]);
    }
    inline void operator/=(const Vector<width> &other) {
        for (unsigned i = 0; i < NUM_REGS; i++)
            v[i] = _mm256_div_ps(v[i], other.v[i]);
    }
    
    inline Vector<width> operator+(const Scalar &other) {
        Vector<width> result;
        for (unsigned i = 0; i < NUM_REGS; i++)
            result.v[i] = _mm256_add_ps(v[i], other.v);
        return result;
    }
    inline Vector<width> operator-(const Scalar &other) {
        Vector<width> result;
        for (unsigned i = 0; i < NUM_REGS; i++)
            result.v[i] = _mm256_sub_ps(v[i], other.v);
        return result;
    }
    inline Vector<width> operator*(const Scalar &other) {
        Vector<width> result;
        for (unsigned i = 0; i < NUM_REGS; i++)
            result.v[i] = _mm256_mul_ps(v[i], other.v);
        return result;
    }
    inline Vector<width> operator/(const Scalar &other) {
        Vector<width> result;
        for (unsigned i = 0; i < NUM_REGS; i++)
            result.v[i] = _mm256_div_ps(v[i], other.v);
        return result;
    }
    
    inline void operator+=(const Scalar &other) {
        for (unsigned i = 0; i < NUM_REGS; i++)
            v[i] = _mm256_add_ps(v[i], other.v);
    }
    inline void operator-=(const Scalar &other) {
        for (unsigned i = 0; i < NUM_REGS; i++)
            v[i] = _mm256_sub_ps(v[i], other.v);
    }
    inline void operator*=(const Scalar &other) {
        for (unsigned i = 0; i < NUM_REGS; i++)
            v[i] = _mm256_mul_ps(v[i], other.v);
    }
    inline void operator/=(const Scalar &other) {
        for (unsigned i = 0; i < NUM_REGS; i++)
            v[i] = _mm256_div_ps(v[i], other.v);
    }
    
    inline void storeHorizontalSum(float *dst) const {
        __m256 sumFull = v[0];
        for (unsigned i = 1; i < NUM_REGS; i++)
            sumFull = _mm256_add_ps(sumFull, v[i]);
        __m128 sumHalf = _mm_add_ps(_mm256_extractf128_ps(sumFull, 0), _mm256_extractf128_ps(sumFull, 1));
        sumHalf = _mm_hadd_ps(sumHalf, sumHalf);
        sumHalf = _mm_hadd_ps(sumHalf, sumHalf);
        _mm_store_ss(dst, sumHalf);
    }
    
    __m256 v[NUM_REGS];
};

template<unsigned width>
inline Vector<width> max(const Vector<width> &lhs, const Vector<width> &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        result.v[i] = _mm256_max_ps(lhs.v[i], rhs.v[i]);
    return result;
}

template<unsigned width>
inline Vector<width> min(const Vector<width> &lhs, const Vector<width> &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        result.v[i] = _mm256_min_ps(lhs.v[i], rhs.v[i]);
    return result;
}

template<unsigned width>
inline Vector<width> max(const Vector<width> &lhs, const Scalar &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        result.v[i] = _mm256_max_ps(lhs.v[i], rhs.v);
    return result;
}

template<unsigned width>
inline Vector<width> min(const Vector<width> &lhs, const Scalar &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        result.v[i] = _mm256_min_ps(lhs.v[i], rhs.v);
    return result;
}


template<unsigned width>
inline Vector<width> operator<(const Vector<width> &lhs, const Vector<width> &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        result.v[i] = _mm256_cmp_ps(lhs.v[i], rhs.v[i], _CMP_LT_OS);
    return result;
}
template<unsigned width>
inline Vector<width> operator<=(const Vector<width> &lhs, const Vector<width> &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        result.v[i] = _mm256_cmp_ps(lhs.v[i], rhs.v[i], _CMP_LE_OS);
    return result;
}
template<unsigned width>
inline Vector<width> operator>(const Vector<width> &lhs, const Vector<width> &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        result.v[i] = _mm256_cmp_ps(lhs.v[i], rhs.v[i], _CMP_GT_OS);
    return result;
}
template<unsigned width>
inline Vector<width> operator>=(const Vector<width> &lhs, const Vector<width> &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        result.v[i] = _mm256_cmp_ps(lhs.v[i], rhs.v[i], _CMP_GE_OS);
    return result;
}
template<unsigned width>
inline Vector<width> operator==(const Vector<width> &lhs, const Vector<width> &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        result.v[i] = _mm256_cmp_ps(lhs.v[i], rhs.v[i], _CMP_EQ_OS);
    return result;
}
template<unsigned width>
inline Vector<width> operator!=(const Vector<width> &lhs, const Vector<width> &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        result.v[i] = _mm256_cmp_ps(lhs.v[i], rhs.v[i], _CMP_NEQ_OS);
    return result;
}


template<unsigned width>
inline Vector<width> operator<(const Vector<width> &lhs, const Scalar &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        result.v[i] = _mm256_cmp_ps(lhs.v[i], rhs.v, _CMP_LT_OS);
    return result;
}
template<unsigned width>
inline Vector<width> operator<=(const Vector<width> &lhs, const Scalar &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        result.v[i] = _mm256_cmp_ps(lhs.v[i], rhs.v, _CMP_LE_OS);
    return result;
}
template<unsigned width>
inline Vector<width> operator>(const Vector<width> &lhs, const Scalar &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        result.v[i] = _mm256_cmp_ps(lhs.v[i], rhs.v, _CMP_GT_OS);
    return result;
}
template<unsigned width>
inline Vector<width> operator>=(const Vector<width> &lhs, const Scalar &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        result.v[i] = _mm256_cmp_ps(lhs.v[i], rhs.v, _CMP_GE_OS);
    return result;
}
template<unsigned width>
inline Vector<width> operator==(const Vector<width> &lhs, const Scalar &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        result.v[i] = _mm256_cmp_ps(lhs.v[i], rhs.v, _CMP_EQ_OS);
    return result;
}
template<unsigned width>
inline Vector<width> operator!=(const Vector<width> &lhs, const Scalar &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        result.v[i] = _mm256_cmp_ps(lhs.v[i], rhs.v, _CMP_NEQ_OS);
    return result;
}


template<unsigned width>
inline Vector<width> select(const Vector<width> &mask, const Vector<width> &valueTrue, const Vector<width> &valueFalse) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        result.v[i] = _mm256_or_ps(_mm256_and_ps(mask.v[i], valueTrue.v[i]), _mm256_andnot_ps(mask.v[i], valueFalse.v[i]));
    return result;
}

template<unsigned width>
inline Vector<width> selectOrZero(const Vector<width> &mask, const Vector<width> &valueTrue) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        result.v[i] = _mm256_and_ps(mask.v[i], valueTrue.v[i]);
    return result;
}

#elif defined(__SSE__) && defined(DIP6_USE_SSE)

#warning "No AVX, falling back to SSE"

enum {
    SIMD_WIDTH = 4,
};
    
struct Scalar
{
    Scalar() = default;
    Scalar(const float *src) { load(src); }
    inline void load(const float *src) { v = _mm_set1_ps(*src); }
    Scalar(InitZeroEnum) { setZero(); }
    inline void setZero() { v = _mm_setzero_ps(); }    
    
    __m128 v;
};

template<unsigned width>
struct Vector 
{
    enum {
        NUM_LANES = width,
        NUM_REGS = width / SIMD_WIDTH
    };
    
    Vector() = default;
    Vector(const float *src) { load(src); }
    Vector(const Scalar &v) { (*this) = v; }
    Vector(InitZeroEnum) { setZero(); }
    
    inline void load(const float *src) {
        for (unsigned i = 0; i < NUM_REGS; i++)
            v[i] = _mm_load_ps(src + i*SIMD_WIDTH);
    }
    inline void store(float *dst) const {
        for (unsigned i = 0; i < NUM_REGS; i++)
            _mm_store_ps(dst + i*SIMD_WIDTH, v[i]);
    }
    
    inline void setZero() {
        for (unsigned i = 0; i < NUM_REGS; i++)
            v[i] = _mm_setzero_ps();
    }
    
    inline void operator=(const Vector<width> &other) {
        for (unsigned i = 0; i < NUM_REGS; i++)
            v[i] = other.v[i];
    }
    inline void operator=(const Scalar &other) {
        for (unsigned i = 0; i < NUM_REGS; i++)
            v[i] = other.v;
    }
    
    inline Vector<width> operator+(const Vector<width> &other) {
        Vector<width> result;
        for (unsigned i = 0; i < NUM_REGS; i++)
            result.v[i] = _mm_add_ps(v[i], other.v[i]);
        return result;
    }
    inline Vector<width> operator-(const Vector<width> &other) {
        Vector<width> result;
        for (unsigned i = 0; i < NUM_REGS; i++)
            result.v[i] = _mm_sub_ps(v[i], other.v[i]);
        return result;
    }
    inline Vector<width> operator*(const Vector<width> &other) {
        Vector<width> result;
        for (unsigned i = 0; i < NUM_REGS; i++)
            result.v[i] = _mm_mul_ps(v[i], other.v[i]);
        return result;
    }
    inline Vector<width> operator/(const Vector<width> &other) {
        Vector<width> result;
        for (unsigned i = 0; i < NUM_REGS; i++)
            result.v[i] = _mm_div_ps(v[i], other.v[i]);
        return result;
    }
    
    inline void operator+=(const Vector<width> &other) {
        for (unsigned i = 0; i < NUM_REGS; i++)
            v[i] = _mm_add_ps(v[i], other.v[i]);
    }
    inline void operator-=(const Vector<width> &other) {
        for (unsigned i = 0; i < NUM_REGS; i++)
            v[i] = _mm_sub_ps(v[i], other.v[i]);
    }
    inline void operator*=(const Vector<width> &other) {
        for (unsigned i = 0; i < NUM_REGS; i++)
            v[i] = _mm_mul_ps(v[i], other.v[i]);
    }
    inline void operator/=(const Vector<width> &other) {
        for (unsigned i = 0; i < NUM_REGS; i++)
            v[i] = _mm_div_ps(v[i], other.v[i]);
    }
    
    inline Vector<width> operator+(const Scalar &other) {
        Vector<width> result;
        for (unsigned i = 0; i < NUM_REGS; i++)
            result.v[i] = _mm_add_ps(v[i], other.v);
        return result;
    }
    inline Vector<width> operator-(const Scalar &other) {
        Vector<width> result;
        for (unsigned i = 0; i < NUM_REGS; i++)
            result.v[i] = _mm_sub_ps(v[i], other.v);
        return result;
    }
    inline Vector<width> operator*(const Scalar &other) {
        Vector<width> result;
        for (unsigned i = 0; i < NUM_REGS; i++)
            result.v[i] = _mm_mul_ps(v[i], other.v);
        return result;
    }
    inline Vector<width> operator/(const Scalar &other) {
        Vector<width> result;
        for (unsigned i = 0; i < NUM_REGS; i++)
            result.v[i] = _mm_div_ps(v[i], other.v);
        return result;
    }
    
    inline void operator+=(const Scalar &other) {
        for (unsigned i = 0; i < NUM_REGS; i++)
            v[i] = _mm_add_ps(v[i], other.v);
    }
    inline void operator-=(const Scalar &other) {
        for (unsigned i = 0; i < NUM_REGS; i++)
            v[i] = _mm_sub_ps(v[i], other.v);
    }
    inline void operator*=(const Scalar &other) {
        for (unsigned i = 0; i < NUM_REGS; i++)
            v[i] = _mm_mul_ps(v[i], other.v);
    }
    inline void operator/=(const Scalar &other) {
        for (unsigned i = 0; i < NUM_REGS; i++)
            v[i] = _mm_div_ps(v[i], other.v);
    }
    
    inline void storeHorizontalSum(float *dst) const {
        __m128 sum = v[0];
        for (unsigned i = 1; i < NUM_REGS; i++)
            sum = _mm_add_ps(sum, v[i]);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        _mm_store_ss(dst, sum);
    }
    
    __m128 v[NUM_REGS];
};


template<unsigned width>
inline Vector<width> max(const Vector<width> &lhs, const Vector<width> &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        result.v[i] = _mm_max_ps(lhs.v[i], rhs.v[i]);
    return result;
}

template<unsigned width>
inline Vector<width> min(const Vector<width> &lhs, const Vector<width> &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        result.v[i] = _mm_min_ps(lhs.v[i], rhs.v[i]);
    return result;
}

template<unsigned width>
inline Vector<width> max(const Vector<width> &lhs, const Scalar &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        result.v[i] = _mm_max_ps(lhs.v[i], rhs.v);
    return result;
}

template<unsigned width>
inline Vector<width> min(const Vector<width> &lhs, const Scalar &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        result.v[i] = _mm_min_ps(lhs.v[i], rhs.v);
    return result;
}


template<unsigned width>
inline Vector<width> operator<(const Vector<width> &lhs, const Vector<width> &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        result.v[i] = _mm_cmplt_ps(lhs.v[i], rhs.v[i]);
    return result;
}
template<unsigned width>
inline Vector<width> operator<=(const Vector<width> &lhs, const Vector<width> &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        result.v[i] = _mm_cmple_ps(lhs.v[i], rhs.v[i]);
    return result;
}
template<unsigned width>
inline Vector<width> operator>(const Vector<width> &lhs, const Vector<width> &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        result.v[i] = _mm_cmpgt_ps(lhs.v[i], rhs.v[i]);
    return result;
}
template<unsigned width>
inline Vector<width> operator>=(const Vector<width> &lhs, const Vector<width> &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        result.v[i] = _mm_cmpge_ps(lhs.v[i], rhs.v[i]);
    return result;
}
template<unsigned width>
inline Vector<width> operator==(const Vector<width> &lhs, const Vector<width> &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        result.v[i] = _mm_cmpeq_ps(lhs.v[i], rhs.v[i]);
    return result;
}
template<unsigned width>
inline Vector<width> operator!=(const Vector<width> &lhs, const Vector<width> &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        result.v[i] = _mm_cmpneq_ps(lhs.v[i], rhs.v[i]);
    return result;
}


template<unsigned width>
inline Vector<width> operator<(const Vector<width> &lhs, const Scalar &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        result.v[i] = _mm_cmplt_ps(lhs.v[i], rhs.v);
    return result;
}
template<unsigned width>
inline Vector<width> operator<=(const Vector<width> &lhs, const Scalar &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        result.v[i] = _mm_cmple_ps(lhs.v[i], rhs.v);
    return result;
}
template<unsigned width>
inline Vector<width> operator>(const Vector<width> &lhs, const Scalar &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        result.v[i] = _mm_cmpgt_ps(lhs.v[i], rhs.v);
    return result;
}
template<unsigned width>
inline Vector<width> operator>=(const Vector<width> &lhs, const Scalar &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        result.v[i] = _mm_cmpge_ps(lhs.v[i], rhs.v);
    return result;
}
template<unsigned width>
inline Vector<width> operator==(const Vector<width> &lhs, const Scalar &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        result.v[i] = _mm_cmpeq_ps(lhs.v[i], rhs.v);
    return result;
}
template<unsigned width>
inline Vector<width> operator!=(const Vector<width> &lhs, const Scalar &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        result.v[i] = _mm_cmpneq_ps(lhs.v[i], rhs.v);
    return result;
}


template<unsigned width>
inline Vector<width> select(const Vector<width> &mask, const Vector<width> &valueTrue, const Vector<width> &valueFalse) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        result.v[i] = _mm_or_ps(_mm_and_ps(mask.v[i], valueTrue.v[i]), _mm_andnot_ps(mask.v[i], valueFalse.v[i]));
    return result;
}

template<unsigned width>
inline Vector<width> selectOrZero(const Vector<width> &mask, const Vector<width> &valueTrue) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        result.v[i] = _mm_and_ps(mask.v[i], valueTrue.v[i]);
    return result;
}



#else

#warning "No AVX or SSE, falling back to scalar floats"

enum {
    SIMD_WIDTH = 1,
};
    
struct Scalar
{
    Scalar() = default;
    Scalar(const float *src) { load(src); }
    inline void load(const float *src) { v = *src; }
    Scalar(InitZeroEnum) { setZero(); }
    inline void setZero() { v = 0.0f; }
    
    float v;
};

template<unsigned width>
struct Vector 
{
    enum {
        NUM_LANES = width,
        NUM_REGS = width / SIMD_WIDTH
    };
    
    Vector() = default;
    Vector(const float *src) { load(src); }
    Vector(const Scalar &v) { (*this) = v; }
    Vector(InitZeroEnum) { setZero(); }
    
    inline void load(const float *src) {
        for (unsigned i = 0; i < NUM_REGS; i++)
            v[i] = src[i*SIMD_WIDTH];
    }
    inline void store(float *dst) const {
        for (unsigned i = 0; i < NUM_REGS; i++)
            dst[i*SIMD_WIDTH] = v[i];
    }
    
    inline void setZero() {
        for (unsigned i = 0; i < NUM_REGS; i++)
            v[i] = 0.0f;
    }
    
    inline void operator=(const Vector<width> &other) {
        for (unsigned i = 0; i < NUM_REGS; i++)
            v[i] = other.v[i];
    }
    inline void operator=(const Scalar &other) {
        for (unsigned i = 0; i < NUM_REGS; i++)
            v[i] = other.v;
    }
    
    inline Vector<width> operator+(const Vector<width> &other) {
        Vector<width> result;
        for (unsigned i = 0; i < NUM_REGS; i++)
            result.v[i] = v[i] + other.v[i];
        return result;
    }
    inline Vector<width> operator-(const Vector<width> &other) {
        Vector<width> result;
        for (unsigned i = 0; i < NUM_REGS; i++)
            result.v[i] = v[i] - other.v[i];
        return result;
    }
    inline Vector<width> operator*(const Vector<width> &other) {
        Vector<width> result;
        for (unsigned i = 0; i < NUM_REGS; i++)
            result.v[i] = v[i] * other.v[i];
        return result;
    }
    inline Vector<width> operator/(const Vector<width> &other) {
        Vector<width> result;
        for (unsigned i = 0; i < NUM_REGS; i++)
            result.v[i] = v[i] / other.v[i];
        return result;
    }
    
    inline void operator+=(const Vector<width> &other) {
        for (unsigned i = 0; i < NUM_REGS; i++)
            v[i] += other.v[i];
    }
    inline void operator-=(const Vector<width> &other) {
        for (unsigned i = 0; i < NUM_REGS; i++)
            v[i] -= other.v[i];
    }
    inline void operator*=(const Vector<width> &other) {
        for (unsigned i = 0; i < NUM_REGS; i++)
            v[i] *= other.v[i];
    }
    inline void operator/=(const Vector<width> &other) {
        for (unsigned i = 0; i < NUM_REGS; i++)
            v[i] /= other.v[i];
    }
    
    inline Vector<width> operator+(const Scalar &other) {
        Vector<width> result;
        for (unsigned i = 0; i < NUM_REGS; i++)
            result.v[i] = v[i] + other.v;
        return result;
    }
    inline Vector<width> operator-(const Scalar &other) {
        Vector<width> result;
        for (unsigned i = 0; i < NUM_REGS; i++)
            result.v[i] = v[i] - other.v;
        return result;
    }
    inline Vector<width> operator*(const Scalar &other) {
        Vector<width> result;
        for (unsigned i = 0; i < NUM_REGS; i++)
            result.v[i] = v[i] * other.v;
        return result;
    }
    inline Vector<width> operator/(const Scalar &other) {
        Vector<width> result;
        for (unsigned i = 0; i < NUM_REGS; i++)
            result.v[i] = v[i] / other.v;
        return result;
    }
    
    inline void operator+=(const Scalar &other) {
        for (unsigned i = 0; i < NUM_REGS; i++)
            v[i] += other.v;
    }
    inline void operator-=(const Scalar &other) {
        for (unsigned i = 0; i < NUM_REGS; i++)
            v[i] -= other.v;
    }
    inline void operator*=(const Scalar &other) {
        for (unsigned i = 0; i < NUM_REGS; i++)
            v[i] *= other.v;
    }
    inline void operator/=(const Scalar &other) {
        for (unsigned i = 0; i < NUM_REGS; i++)
            v[i] /= other.v;
    }
    
    inline void storeHorizontalSum(float *dst) const {
        float sum = v[0];
        for (unsigned i = 1; i < NUM_REGS; i++)
            sum  += v[i];
        *dst = sum;
    }
    
    float v[NUM_REGS];
};


template<unsigned width>
inline Vector<width> max(const Vector<width> &lhs, const Vector<width> &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        result.v[i] = std::max(lhs.v[i], rhs.v[i]);
    return result;
}

template<unsigned width>
inline Vector<width> min(const Vector<width> &lhs, const Vector<width> &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        result.v[i] = std::min(lhs.v[i], rhs.v[i]);
    return result;
}

template<unsigned width>
inline Vector<width> max(const Vector<width> &lhs, const Scalar &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        result.v[i] = std::max(lhs.v[i], rhs.v);
    return result;
}

template<unsigned width>
inline Vector<width> min(const Vector<width> &lhs, const Scalar &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        result.v[i] = std::min(lhs.v[i], rhs.v);
    return result;
}


template<unsigned width>
inline Vector<width> operator<(const Vector<width> &lhs, const Vector<width> &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        reinterpret_cast<std::uint32_t&>(result.v[i]) = lhs.v[i] < rhs.v[i]?0xFFFFFFFF:0x00000000;
    return result;
}
template<unsigned width>
inline Vector<width> operator<=(const Vector<width> &lhs, const Vector<width> &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        reinterpret_cast<std::uint32_t&>(result.v[i]) = lhs.v[i] <= rhs.v[i]?0xFFFFFFFF:0x00000000;
    return result;
}
template<unsigned width>
inline Vector<width> operator>(const Vector<width> &lhs, const Vector<width> &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        reinterpret_cast<std::uint32_t&>(result.v[i]) = lhs.v[i] > rhs.v[i]?0xFFFFFFFF:0x00000000;
    return result;
}
template<unsigned width>
inline Vector<width> operator>=(const Vector<width> &lhs, const Vector<width> &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        reinterpret_cast<std::uint32_t&>(result.v[i]) = lhs.v[i] >= rhs.v[i]?0xFFFFFFFF:0x00000000;
    return result;
}
template<unsigned width>
inline Vector<width> operator==(const Vector<width> &lhs, const Vector<width> &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        reinterpret_cast<std::uint32_t&>(result.v[i]) = lhs.v[i] == rhs.v[i]?0xFFFFFFFF:0x00000000;
    return result;
}
template<unsigned width>
inline Vector<width> operator!=(const Vector<width> &lhs, const Vector<width> &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        reinterpret_cast<std::uint32_t&>(result.v[i]) = lhs.v[i] != rhs.v[i]?0xFFFFFFFF:0x00000000;
    return result;
}


template<unsigned width>
inline Vector<width> operator<(const Vector<width> &lhs, const Scalar &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        reinterpret_cast<std::uint32_t&>(result.v[i]) = lhs.v[i] < rhs.v?0xFFFFFFFF:0x00000000;
    return result;
}
template<unsigned width>
inline Vector<width> operator<=(const Vector<width> &lhs, const Scalar &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        reinterpret_cast<std::uint32_t&>(result.v[i]) = lhs.v[i] <= rhs.v?0xFFFFFFFF:0x00000000;
    return result;
}
template<unsigned width>
inline Vector<width> operator>(const Vector<width> &lhs, const Scalar &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        reinterpret_cast<std::uint32_t&>(result.v[i]) = lhs.v[i] > rhs.v?0xFFFFFFFF:0x00000000;
    return result;
}
template<unsigned width>
inline Vector<width> operator>=(const Vector<width> &lhs, const Scalar &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        reinterpret_cast<std::uint32_t&>(result.v[i]) = lhs.v[i] >= rhs.v?0xFFFFFFFF:0x00000000;
    return result;
}
template<unsigned width>
inline Vector<width> operator==(const Vector<width> &lhs, const Scalar &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        reinterpret_cast<std::uint32_t&>(result.v[i]) = lhs.v[i] == rhs.v?0xFFFFFFFF:0x00000000;
    return result;
}
template<unsigned width>
inline Vector<width> operator!=(const Vector<width> &lhs, const Scalar &rhs) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        reinterpret_cast<std::uint32_t&>(result.v[i]) = lhs.v[i] != rhs.v?0xFFFFFFFF:0x00000000;
    return result;
}


template<unsigned width>
inline Vector<width> select(const Vector<width> &mask, const Vector<width> &valueTrue, const Vector<width> &valueFalse) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        reinterpret_cast<std::uint32_t&>(result.v[i]) = 
                (reinterpret_cast<const std::uint32_t&>(valueTrue.v[i]) & reinterpret_cast<const std::uint32_t&>(mask.v[i])) |
                (reinterpret_cast<const std::uint32_t&>(valueFalse.v[i]) & ~reinterpret_cast<const std::uint32_t&>(mask.v[i]));
    return result;
}

template<unsigned width>
inline Vector<width> selectOrZero(const Vector<width> &mask, const Vector<width> &valueTrue) {
    Vector<width> result;
    for (unsigned i = 0; i < Vector<width>::NUM_REGS; i++)
        reinterpret_cast<std::uint32_t&>(result.v[i]) = 
                reinterpret_cast<const std::uint32_t&>(valueTrue.v[i]) & reinterpret_cast<const std::uint32_t&>(mask.v[i]);
    return result;
}


#endif
    
    
}
}

#endif // SIMD_H
