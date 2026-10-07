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

inline int32x4_t dot16_acc(int32x4_t acc, int8x16_t w, int8x16_t x) { return vdotq_s32(acc, w, x); }

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

// Four rows at once from the interleaved layout: each sdot covers 8 elements
// of two rows, so one block of four rows takes 8 sdots and one pairwise add
// yields the four row sums.
void matvec_q4_0x4_dotprod(const void* w, std::size_t rows, std::size_t cols, const void* x, float* out) {
    const auto* weights = static_cast<const BlockQ4_0x4*>(w);
    const auto* activations = static_cast<const BlockQ8_0*>(x);
    const std::size_t blocks = cols / kBlockSize;
    const uint8x16_t low_mask = vdupq_n_u8(0x0F);
    const int8x16_t eight = vdupq_n_s8(8);

    for (std::size_t g = 0; g < rows / 4; ++g) {
        const BlockQ4_0x4* group = weights + g * blocks;
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (std::size_t b = 0; b < blocks; ++b) {
            const BlockQ8_0& xb = activations[b];
            int8x16_t a[4][2];  // [vector k][low, high]
            for (int k = 0; k < 4; ++k) {
                const uint8x16_t v = vld1q_u8(group[b].qs + 16 * k);
                a[k][0] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, low_mask)), eight);
                a[k][1] = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v, 4)), eight);
            }
            // x chunk c duplicated into both halves, to multiply two rows' chunk c at once.
            int8x16_t xc[4];
            for (int c = 0; c < 4; ++c) xc[c] = vcombine_s8(vld1_s8(xb.qs + 8 * c), vld1_s8(xb.qs + 8 * c));
            // Vector k: rows 2(k/2), 2(k/2)+1; low nibbles = chunk k%2, high = chunk 2 + k%2.
            int32x4_t s01 = vdotq_s32(vdupq_n_s32(0), a[0][0], xc[0]);
            s01 = vdotq_s32(s01, a[1][0], xc[1]);
            s01 = vdotq_s32(s01, a[0][1], xc[2]);
            s01 = vdotq_s32(s01, a[1][1], xc[3]);
            int32x4_t s23 = vdotq_s32(vdupq_n_s32(0), a[2][0], xc[0]);
            s23 = vdotq_s32(s23, a[3][0], xc[1]);
            s23 = vdotq_s32(s23, a[2][1], xc[2]);
            s23 = vdotq_s32(s23, a[3][1], xc[3]);
            // Lanes of s01: [r0 half, r0 half, r1 half, r1 half]; pairwise add gives [r0, r1, r2, r3].
            const int32x4_t sums = vpaddq_s32(s01, s23);
            const float32x4_t scales = vmulq_n_f32(vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(group[b].d))),
                                                   vgetq_lane_f32(neon::half4_to_float(xb.d), 0));
            acc = vmlaq_f32(acc, vcvtq_f32_s32(sums), scales);
        }
        vst1q_f32(out + 4 * g, acc);
    }
}

void matvec_tq2_0_dotprod(const void* w, std::size_t rows, std::size_t cols, const void* x, float* out) {
    neon::matmul_k(static_cast<const BlockTQ2_0*>(w), rows, cols, static_cast<const BlockQ8_K*>(x), 1, out,
                   [](const BlockTQ2_0* row, const BlockQ8_K* const* xs, std::size_t n, std::size_t blocks, float* o) {
                       neon::dot_rows_tq2_0(row, xs, n, blocks, o, dot16_acc);
                   });
}

void matmul_tq2_0_dotprod(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out) {
    neon::matmul_k(static_cast<const BlockTQ2_0*>(w), rows, cols, static_cast<const BlockQ8_K*>(x), n, out,
                   [](const BlockTQ2_0* row, const BlockQ8_K* const* xs, std::size_t count, std::size_t blocks, float* o) {
                       neon::dot_rows_tq2_0(row, xs, count, blocks, o, dot16_acc);
                   });
}

// Four ternary rows at once from the interleaved layout. Each vector of the
// block unpacks into four [row a chunk | row b chunk] vectors (one per bit
// pair), each dotted with the matching activation chunk duplicated into both
// halves. sum((q - 1) x) = sum(q x) - sum(x).
namespace {

template <int kShift>
inline int8x16_t ternary_layer(uint8x16_t packed) {
    if constexpr (kShift == 0) return vreinterpretq_s8_u8(vandq_u8(packed, vdupq_n_u8(3)));
    else if constexpr (kShift == 6) return vreinterpretq_s8_u8(vshrq_n_u8(packed, 6));
    else return vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(packed, kShift), vdupq_n_u8(3)));
}

template <int kLayer>
inline void ternary_dot_layer(uint8x16_t w0, uint8x16_t w1, const std::int8_t* x_chunk, int32x4_t& s01, int32x4_t& s23) {
    const int8x8_t xc = vld1_s8(x_chunk);
    const int8x16_t xdup = vcombine_s8(xc, xc);
    s01 = vdotq_s32(s01, ternary_layer<2 * kLayer>(w0), xdup);
    s23 = vdotq_s32(s23, ternary_layer<2 * kLayer>(w1), xdup);
}

}  // namespace

void matvec_tq2_0x4_dotprod(const void* w, std::size_t rows, std::size_t cols, const void* x, float* out) {
    const auto* weights = static_cast<const BlockTQ2_0x4*>(w);
    const auto* activations = static_cast<const BlockQ8_K*>(x);
    const std::size_t blocks = cols / kSuperBlockSize;

    for (std::size_t g = 0; g < rows / 4; ++g) {
        const BlockTQ2_0x4* group = weights + g * blocks;
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (std::size_t b = 0; b < blocks; ++b) {
            const BlockQ8_K& xb = activations[b];
            int32x4_t s01 = vdupq_n_s32(0), s23 = vdupq_n_s32(0);
            for (int v = 0; v < 8; ++v) {
                const uint8x16_t w0 = vld1q_u8(group[b].qs + 16 * v);
                const uint8x16_t w1 = vld1q_u8(group[b].qs + 128 + 16 * v);
                const std::int8_t* xc = xb.qs + 32 * v;  // chunks 4v .. 4v+3
                ternary_dot_layer<0>(w0, w1, xc, s01, s23);
                ternary_dot_layer<1>(w0, w1, xc + 8, s01, s23);
                ternary_dot_layer<2>(w0, w1, xc + 16, s01, s23);
                ternary_dot_layer<3>(w0, w1, xc + 24, s01, s23);
            }
            // Lanes [r0 half, r0 half, r1 half, r1 half] -> [r0, r1, r2, r3], minus sum(x) per row.
            const int16x8_t bsums = vaddq_s16(vld1q_s16(xb.bsums), vld1q_s16(xb.bsums + 8));
            const int32x4_t sums = vsubq_s32(vpaddq_s32(s01, s23), vdupq_n_s32(vaddlvq_s16(bsums)));
            const float32x4_t scales = vmulq_n_f32(vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(group[b].d))), xb.d);
            acc = vmlaq_f32(acc, vcvtq_f32_s32(sums), scales);
        }
        vst1q_f32(out + 4 * g, acc);
    }
}

}  // namespace brisk::kernels
