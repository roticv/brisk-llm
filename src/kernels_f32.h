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

// Attention's score step: scores[q][p] = scale * dot(queries[q], keys[p]) for
// `nq` query vectors against `np` key rows (stride key_stride), all of `dim`
// floats (a multiple of 4). Query q's row of scores is written to scores[q].
// Register-blocked 4x4 on NEON.
void scores_f32(const float* const* queries, std::size_t nq, const float* keys, std::size_t key_stride, std::size_t np,
                std::size_t dim, float scale, float* const* scores);

// Attention's value step for `heads` query heads sharing one cached value
// head: out[h] = sum over p < count of weights[h * stride + p] * v[p], where
// v[p] is at values + p * value_stride and all vectors have `dim` floats
// (a multiple of 32). `out` is overwritten.
void weighted_sum_f32(const float* values, std::size_t value_stride, std::size_t count, const float* weights,
                      std::size_t stride, std::size_t heads, std::size_t dim, float* out);

}  // namespace brisk::kernels
