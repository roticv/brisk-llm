// ARMv8.2 dot-product kernels (sdot). Compiled with +dotprod; only called when
// cpu_features() reports support.

#include "kernels.h"
#include "kernels_neon_common.h"

namespace brisk::kernels {

namespace {

inline int32x4_t dot_block(int8x16_t w_low, int8x16_t w_high, const BlockQ8_0& x) {
    const int32x4_t acc = vdotq_s32(vdupq_n_s32(0), w_low, vld1q_s8(x.qs));
    return vdotq_s32(acc, w_high, vld1q_s8(x.qs + 16));
}

inline int32x4_t dot16(int8x16_t w, int8x16_t x) { return vdotq_s32(vdupq_n_s32(0), w, x); }

}  // namespace

void matvec_q4_0_dotprod(const void* w, std::size_t rows, std::size_t cols, const void* x, float* out) {
    const auto* weights = static_cast<const BlockQ4_0*>(w);
    const auto* activations = static_cast<const BlockQ8_0*>(x);
    const std::size_t blocks = cols / kBlockSize;
    for (std::size_t r = 0; r < rows; ++r) out[r] = neon::dot_row(weights + r * blocks, activations, blocks, dot_block);
}

void matmul_q4_0_dotprod(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out) {
    neon::matmul(static_cast<const BlockQ4_0*>(w), rows, cols, static_cast<const BlockQ8_0*>(x), n, out, dot_block);
}

void matvec_q4_k_dotprod(const void* w, std::size_t rows, std::size_t cols, const void* x, float* out) {
    neon::matmul_k(static_cast<const BlockQ4_K*>(w), rows, cols, static_cast<const BlockQ8_K*>(x), 1, out,
                   [](const BlockQ4_K* row, const BlockQ8_K* const* xs, std::size_t n, std::size_t blocks, float* o) {
                       neon::dot_rows_q4_k(row, xs, n, blocks, o, dot16);
                   });
}

void matmul_q4_k_dotprod(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out) {
    neon::matmul_k(static_cast<const BlockQ4_K*>(w), rows, cols, static_cast<const BlockQ8_K*>(x), n, out,
                   [](const BlockQ4_K* row, const BlockQ8_K* const* xs, std::size_t count, std::size_t blocks, float* o) {
                       neon::dot_rows_q4_k(row, xs, count, blocks, o, dot16);
                   });
}

void matvec_q6_k_dotprod(const void* w, std::size_t rows, std::size_t cols, const void* x, float* out) {
    neon::matmul_k(static_cast<const BlockQ6_K*>(w), rows, cols, static_cast<const BlockQ8_K*>(x), 1, out,
                   [](const BlockQ6_K* row, const BlockQ8_K* const* xs, std::size_t n, std::size_t blocks, float* o) {
                       neon::dot_rows_q6_k(row, xs, n, blocks, o, dot16);
                   });
}

void matmul_q6_k_dotprod(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out) {
    neon::matmul_k(static_cast<const BlockQ6_K*>(w), rows, cols, static_cast<const BlockQ8_K*>(x), n, out,
                   [](const BlockQ6_K* row, const BlockQ8_K* const* xs, std::size_t count, std::size_t blocks, float* o) {
                       neon::dot_rows_q6_k(row, xs, count, blocks, o, dot16);
                   });
}

}  // namespace brisk::kernels
