#pragma once

#include <cstddef>
#include <span>

// Float vector routines used by attention and the norms: NEON on AArch64,
// plain loops elsewhere.

namespace brisk::kernels {

float dot_f32(const float* a, const float* b, std::size_t n);
// out[i] += scale * x[i]
void axpy_f32(float scale, const float* x, float* out, std::size_t n);
// x[i] = exp(x[i] - max(x)) / sum, in place.
void softmax_f32(std::span<float> x);

// Attention's value step for `heads` query heads sharing one cached value
// head: out[h] = sum over p < count of weights[h * stride + p] * v[p], where
// v[p] is at values + p * value_stride and all vectors have `dim` floats
// (a multiple of 32). `out` is overwritten.
void weighted_sum_f32(const float* values, std::size_t value_stride, std::size_t count, const float* weights,
                      std::size_t stride, std::size_t heads, std::size_t dim, float* out);

}  // namespace brisk::kernels
