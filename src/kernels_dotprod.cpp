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

void matvec_q4_1_dotprod(const void* w, std::size_t rows, std::size_t cols, const void* x, float* out) {
    const auto* weights = static_cast<const BlockQ4_1*>(w);
    const auto* activations = static_cast<const BlockQ8_0*>(x);
    const std::size_t blocks = cols / kBlockSize;
    for (std::size_t r = 0; r < rows; ++r) out[r] = neon::dot_row_q4_1(weights + r * blocks, activations, blocks, dot_block);
}

void matmul_q4_1_dotprod(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out) {
    const auto* weights = static_cast<const BlockQ4_1*>(w);
    const auto* activations = static_cast<const BlockQ8_0*>(x);
    const std::size_t blocks = cols / kBlockSize;
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t t = 0; t < n; ++t) {
            out[t * rows + r] = neon::dot_row_q4_1(weights + r * blocks, activations + t * blocks, blocks, dot_block);
        }
    }
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

// ---- grouped K-quants ------------------------------------------------------

namespace {

// [x chunk c | x chunk c]: 8 activations duplicated, to dot two rows' chunk at once.
inline int8x16_t chunk_dup(const std::int8_t* x, int c) {
    const int8x8_t v = vld1_s8(x + 8 * c);
    return vcombine_s8(v, v);
}

// Four lanes [r0, r1, r2, r3] from two [ra, ra, rb, rb] partials.
inline int32x4_t row_sums(int32x4_t s01, int32x4_t s23) { return vpaddq_s32(s01, s23); }

// Per-row dot of 8 or 16 int16 values with the matching activation sums.
inline std::int32_t dot_s16(const std::int16_t* a, const std::int16_t* b, int n) {
    int32x4_t acc = vmull_s16(vld1_s16(a), vld1_s16(b));
    acc = vmlal_s16(acc, vld1_s16(a + 4), vld1_s16(b + 4));
    if (n == 16) {
        acc = vmlal_s16(acc, vld1_s16(a + 8), vld1_s16(b + 8));
        acc = vmlal_s16(acc, vld1_s16(a + 12), vld1_s16(b + 12));
    }
    return vaddvq_s32(acc);
}

}  // namespace

void matvec_q4_kx4_dotprod(const void* w, std::size_t rows, std::size_t cols, const void* x, float* out) {
    const auto* weights = static_cast<const BlockQ4_Kx4*>(w);
    const auto* activations = static_cast<const BlockQ8_K*>(x);
    const std::size_t blocks = cols / kSuperBlockSize;
    const uint8x16_t low_mask = vdupq_n_u8(0x0F);

    for (std::size_t g = 0; g < rows / 4; ++g) {
        const BlockQ4_Kx4* group = weights + g * blocks;
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (std::size_t b = 0; b < blocks; ++b) {
            const BlockQ4_Kx4& wb = group[b];
            const BlockQ8_K& xb = activations[b];
            std::int16_t scales[4][8], mins[4][8];
            for (int r = 0; r < 4; ++r) {
                for (int sb = 0; sb < 8; ++sb) {
                    std::uint8_t sc, mn;
                    q4_k_scale_min(wb.scales[r], sb, sc, mn);
                    scales[r][sb] = sc;
                    mins[r][sb] = mn;
                }
            }
            // Activation sums per sub-block of 32, from the sums per 16.
            std::int16_t bsum32[8];
            vst1q_s16(bsum32, vpaddq_s16(vld1q_s16(xb.bsums), vld1q_s16(xb.bsums + 8)));

            int32x4_t weighted = vdupq_n_s32(0);  // per row, scaled by the sub-block scales
            for (int sb = 0; sb < 8; ++sb) {
                const std::int8_t* xs = xb.qs + 32 * sb;
                const int8x16_t x0 = chunk_dup(xs, 0), x1 = chunk_dup(xs, 1), x2 = chunk_dup(xs, 2), x3 = chunk_dup(xs, 3);
                int32x4_t partial[2];
                for (int rp = 0; rp < 2; ++rp) {
                    const uint8x16_t v0 = vld1q_u8(wb.qs + 256 * rp + 32 * sb);
                    const uint8x16_t v1 = vld1q_u8(wb.qs + 256 * rp + 32 * sb + 16);
                    int32x4_t sum = vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(vandq_u8(v0, low_mask)), x0);
                    sum = vdotq_s32(sum, vreinterpretq_s8_u8(vandq_u8(v1, low_mask)), x1);
                    sum = vdotq_s32(sum, vreinterpretq_s8_u8(vshrq_n_u8(v0, 4)), x2);
                    partial[rp] = vdotq_s32(sum, vreinterpretq_s8_u8(vshrq_n_u8(v1, 4)), x3);
                }
                const int32x4_t scale = {scales[0][sb], scales[1][sb], scales[2][sb], scales[3][sb]};
                weighted = vmlaq_s32(weighted, row_sums(partial[0], partial[1]), scale);
            }
            const int32x4_t offsets = {dot_s16(mins[0], bsum32, 8), dot_s16(mins[1], bsum32, 8),
                                       dot_s16(mins[2], bsum32, 8), dot_s16(mins[3], bsum32, 8)};
            const float32x4_t d = vmulq_n_f32(vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(wb.d))), xb.d);
            const float32x4_t dmin = vmulq_n_f32(vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(wb.dmin))), xb.d);
            acc = vmlaq_f32(acc, vcvtq_f32_s32(weighted), d);
            acc = vmlsq_f32(acc, vcvtq_f32_s32(offsets), dmin);
        }
        vst1q_f32(out + 4 * g, acc);
    }
}

namespace {

// The eight 6-bit chunk vectors of one row pair's 64-element group, values 0..63.
inline void unpack_q6_kx4_group(const std::uint8_t* low, const std::uint8_t* high, uint8x16_t a[8]) {
    const uint8x16_t low_mask = vdupq_n_u8(0x0F);
    const uint8x16_t two_bits = vdupq_n_u8(0x30);
    for (int v = 0; v < 4; ++v) {
        const uint8x16_t l = vld1q_u8(low + 16 * v);
        a[v] = vandq_u8(l, low_mask);
        a[v + 4] = vshrq_n_u8(l, 4);
    }
    for (int v = 0; v < 2; ++v) {
        const uint8x16_t h = vld1q_u8(high + 16 * v);
        a[4 * v + 0] = vorrq_u8(a[4 * v + 0], vandq_u8(vshlq_n_u8(h, 4), two_bits));
        a[4 * v + 1] = vorrq_u8(a[4 * v + 1], vandq_u8(vshlq_n_u8(h, 2), two_bits));
        a[4 * v + 2] = vorrq_u8(a[4 * v + 2], vandq_u8(h, two_bits));
        a[4 * v + 3] = vorrq_u8(a[4 * v + 3], vandq_u8(vshrq_n_u8(h, 2), two_bits));
    }
}

}  // namespace

void matvec_q6_kx4_dotprod(const void* w, std::size_t rows, std::size_t cols, const void* x, float* out) {
    const auto* weights = static_cast<const BlockQ6_Kx4*>(w);
    const auto* activations = static_cast<const BlockQ8_K*>(x);
    const std::size_t blocks = cols / kSuperBlockSize;

    for (std::size_t g = 0; g < rows / 4; ++g) {
        const BlockQ6_Kx4* group = weights + g * blocks;
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (std::size_t b = 0; b < blocks; ++b) {
            const BlockQ6_Kx4& wb = group[b];
            const BlockQ8_K& xb = activations[b];
            int32x4_t weighted = vdupq_n_s32(0);
            for (int grp = 0; grp < 4; ++grp) {
                uint8x16_t a[2][8];
                unpack_q6_kx4_group(wb.low[0][grp], wb.high[0][grp], a[0]);
                unpack_q6_kx4_group(wb.low[1][grp], wb.high[1][grp], a[1]);
                const std::int8_t* xs = xb.qs + 64 * grp;
                for (int j = 0; j < 4; ++j) {  // sub-blocks of 16 = chunk pairs
                    const int8x16_t x0 = chunk_dup(xs, 2 * j), x1 = chunk_dup(xs, 2 * j + 1);
                    int32x4_t partial[2];
                    for (int rp = 0; rp < 2; ++rp) {
                        const int32x4_t sum = vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(a[rp][2 * j]), x0);
                        partial[rp] = vdotq_s32(sum, vreinterpretq_s8_u8(a[rp][2 * j + 1]), x1);
                    }
                    const int sb = 4 * grp + j;
                    const int32x4_t scale = {wb.scales[0][sb], wb.scales[1][sb], wb.scales[2][sb], wb.scales[3][sb]};
                    weighted = vmlaq_s32(weighted, row_sums(partial[0], partial[1]), scale);
                }
            }
            // sum(scale (q - 32) x) = weighted - 32 sum(scale bsum16)
            std::int16_t scales16[4][16];
            for (int r = 0; r < 4; ++r) {
                for (int sb = 0; sb < 16; ++sb) scales16[r][sb] = wb.scales[r][sb];
            }
            const int32x4_t correction = {dot_s16(scales16[0], xb.bsums, 16), dot_s16(scales16[1], xb.bsums, 16),
                                          dot_s16(scales16[2], xb.bsums, 16), dot_s16(scales16[3], xb.bsums, 16)};
            weighted = vsubq_s32(weighted, vshlq_n_s32(correction, 5));
            const float32x4_t d = vmulq_n_f32(vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(wb.d))), xb.d);
            acc = vmlaq_f32(acc, vcvtq_f32_s32(weighted), d);
        }
        vst1q_f32(out + 4 * g, acc);
    }
}

}  // namespace brisk::kernels
