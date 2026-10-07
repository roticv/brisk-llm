// Baseline NEON (ARMv8.0) kernels: these run on the Raspberry Pi 4's
// Cortex-A72, which has no 8-bit dot-product instruction, so the products are
// widened to 16 bits and summed pairwise.

#include "kernels.h"
#include "kernels_neon_common.h"

namespace brisk::kernels {

namespace {

// The four products summed into each 16-bit lane are at most 8 * 127 each
// (Q4_0 weights lie in [-8, 7]), so they cannot overflow before widening.
inline int32x4_t dot_block(int8x16_t w_low, int8x16_t w_high, const BlockQ8_0& x) {
    const int8x16_t x_low = vld1q_s8(x.qs);
    const int8x16_t x_high = vld1q_s8(x.qs + 16);
    int16x8_t p = vmull_s8(vget_low_s8(w_low), vget_low_s8(x_low));
    p = vmlal_s8(p, vget_high_s8(w_low), vget_high_s8(x_low));
    p = vmlal_s8(p, vget_low_s8(w_high), vget_low_s8(x_high));
    p = vmlal_s8(p, vget_high_s8(w_high), vget_high_s8(x_high));
    return vpaddlq_s16(p);
}

inline int32x4_t dot16(int8x16_t w, int8x16_t x) {
    const int16x8_t p = vmlal_s8(vmull_s8(vget_low_s8(w), vget_low_s8(x)), vget_high_s8(w), vget_high_s8(x));
    return vpaddlq_s16(p);
}

}  // namespace

void matvec_q4_0_neon(const void* w, std::size_t rows, std::size_t cols, const void* x, float* out) {
    const auto* weights = static_cast<const BlockQ4_0*>(w);
    const auto* activations = static_cast<const BlockQ8_0*>(x);
    const std::size_t blocks = cols / kBlockSize;
    for (std::size_t r = 0; r < rows; ++r) out[r] = neon::dot_row(weights + r * blocks, activations, blocks, dot_block);
}

void matmul_q4_0_neon(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out) {
    neon::matmul(static_cast<const BlockQ4_0*>(w), rows, cols, static_cast<const BlockQ8_0*>(x), n, out, dot_block);
}

void matvec_q4_k_neon(const void* w, std::size_t rows, std::size_t cols, const void* x, float* out) {
    neon::matmul_k(static_cast<const BlockQ4_K*>(w), rows, cols, static_cast<const BlockQ8_K*>(x), 1, out,
                   [](const BlockQ4_K* row, const BlockQ8_K* const* xs, std::size_t n, std::size_t blocks, float* o) {
                       neon::dot_rows_q4_k(row, xs, n, blocks, o, dot16);
                   });
}

void matmul_q4_k_neon(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out) {
    neon::matmul_k(static_cast<const BlockQ4_K*>(w), rows, cols, static_cast<const BlockQ8_K*>(x), n, out,
                   [](const BlockQ4_K* row, const BlockQ8_K* const* xs, std::size_t count, std::size_t blocks, float* o) {
                       neon::dot_rows_q4_k(row, xs, count, blocks, o, dot16);
                   });
}

void matvec_q6_k_neon(const void* w, std::size_t rows, std::size_t cols, const void* x, float* out) {
    neon::matmul_k(static_cast<const BlockQ6_K*>(w), rows, cols, static_cast<const BlockQ8_K*>(x), 1, out,
                   [](const BlockQ6_K* row, const BlockQ8_K* const* xs, std::size_t n, std::size_t blocks, float* o) {
                       neon::dot_rows_q6_k(row, xs, n, blocks, o, dot16);
                   });
}

void matmul_q6_k_neon(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out) {
    neon::matmul_k(static_cast<const BlockQ6_K*>(w), rows, cols, static_cast<const BlockQ8_K*>(x), n, out,
                   [](const BlockQ6_K* row, const BlockQ8_K* const* xs, std::size_t count, std::size_t blocks, float* o) {
                       neon::dot_rows_q6_k(row, xs, count, blocks, o, dot16);
                   });
}

}  // namespace brisk::kernels
