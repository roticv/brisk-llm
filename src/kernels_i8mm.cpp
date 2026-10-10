// ARMv8.6 int8 matrix-multiply kernels (smmla). Compiled with +i8mm; only
// called when cpu_features() reports support.
//
// smmla takes a as a 2x8 matrix (two weight rows, 8 elements each), b as an
// 8x2 matrix (two tokens) and accumulates the 2x2 result, so one instruction
// does 32 multiply-adds against sdot's 16. The tile here is 4 rows x 4 tokens,
// which keeps 4 accumulators, 8 weight and 8 activation vectors live.

#include <cstring>

#include "kernels.h"
#include "kernels_neon_common.h"

namespace brisk::kernels {

namespace {

inline int32x4_t dot_block(int8x16_t w_low, int8x16_t w_high, const BlockQ8_0& x) {
    const int32x4_t acc = vdotq_s32(vdupq_n_s32(0), w_low, vld1q_s8(x.qs));
    return vdotq_s32(acc, w_high, vld1q_s8(x.qs + 16));
}

// Rows r and r+1 of a block as four vectors, one per 8-element chunk, each
// holding [row r chunk | row r+1 chunk], the layout smmla wants.
inline void zip_rows(const BlockQ4_0& r0, const BlockQ4_0& r1, int8x16_t a[4]) {
    int8x16_t l0, h0, l1, h1;
    neon::unpack_q4_0(r0, l0, h0);
    neon::unpack_q4_0(r1, l1, h1);
    a[0] = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(l0), vreinterpretq_s64_s8(l1)));
    a[1] = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(l0), vreinterpretq_s64_s8(l1)));
    a[2] = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(h0), vreinterpretq_s64_s8(h1)));
    a[3] = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(h0), vreinterpretq_s64_s8(h1)));
}

// Tokens t and t+1 of a block in the same chunked pair layout.
inline void zip_tokens(const BlockQ8_0& t0, const BlockQ8_0& t1, int8x16_t b[4]) {
    for (int k = 0; k < 4; ++k) b[k] = vcombine_s8(vld1_s8(t0.qs + 8 * k), vld1_s8(t1.qs + 8 * k));
}

// [d0, d0, d1, d1] and [d0, d1, d0, d1]: the row and token scale patterns
// matching smmla's 2x2 output lanes (r0t0, r0t1, r1t0, r1t1).
inline float32x4_t row_scales(std::uint16_t d0, std::uint16_t d1) {
    const float32x4_t d = vcvt_f32_f16(vreinterpret_f16_u16(vset_lane_u16(d1, vdup_n_u16(d0), 1)));
    return vzip1q_f32(d, d);
}
inline float32x4_t token_scales(std::uint16_t d0, std::uint16_t d1) {
    const float32x4_t d = vcvt_f32_f16(vreinterpret_f16_u16(vset_lane_u16(d1, vdup_n_u16(d0), 1)));
    return vcombine_f32(vget_low_f32(d), vget_low_f32(d));
}

inline int32x4_t mmla4(const int8x16_t a[4], const int8x16_t b[4]) {
    int32x4_t acc = vmmlaq_s32(vdupq_n_s32(0), a[0], b[0]);
    acc = vmmlaq_s32(acc, a[1], b[1]);
    acc = vmmlaq_s32(acc, a[2], b[2]);
    return vmmlaq_s32(acc, a[3], b[3]);
}

// 4 weight rows x 4 tokens over a whole row; out[(t + i) * rows + r + j].
void tile_4x4(const BlockQ4_0* w, std::size_t blocks, const BlockQ8_0* x, std::size_t x_stride, float* out,
              std::size_t out_stride) {
    float32x4_t acc[2][2] = {{vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)}, {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)}};
    const BlockQ4_0* rows[4] = {w, w + blocks, w + 2 * blocks, w + 3 * blocks};
    const BlockQ8_0* tokens[4] = {x, x + x_stride, x + 2 * x_stride, x + 3 * x_stride};
    for (std::size_t blk = 0; blk < blocks; ++blk) {
        int8x16_t a[2][4], b[2][4];
        zip_rows(rows[0][blk], rows[1][blk], a[0]);
        zip_rows(rows[2][blk], rows[3][blk], a[1]);
        zip_tokens(tokens[0][blk], tokens[1][blk], b[0]);
        zip_tokens(tokens[2][blk], tokens[3][blk], b[1]);
        const float32x4_t rs[2] = {row_scales(rows[0][blk].d, rows[1][blk].d), row_scales(rows[2][blk].d, rows[3][blk].d)};
        const float32x4_t ts[2] = {token_scales(tokens[0][blk].d, tokens[1][blk].d),
                                   token_scales(tokens[2][blk].d, tokens[3][blk].d)};
        for (int rp = 0; rp < 2; ++rp) {
            for (int tp = 0; tp < 2; ++tp) {
                acc[rp][tp] = vmlaq_f32(acc[rp][tp], vcvtq_f32_s32(mmla4(a[rp], b[tp])), vmulq_f32(rs[rp], ts[tp]));
            }
        }
    }
    for (std::size_t rp = 0; rp < 2; ++rp) {
        for (std::size_t tp = 0; tp < 2; ++tp) {
            // Lanes: (row 2rp, token 2tp), (row 2rp, token 2tp+1), (row 2rp+1, token 2tp), (row 2rp+1, token 2tp+1)
            float* o = out + 2 * tp * out_stride + 2 * rp;
            o[0] = vgetq_lane_f32(acc[rp][tp], 0);
            o[out_stride] = vgetq_lane_f32(acc[rp][tp], 1);
            o[1] = vgetq_lane_f32(acc[rp][tp], 2);
            o[out_stride + 1] = vgetq_lane_f32(acc[rp][tp], 3);
        }
    }
}

}  // namespace

void matmul_q4_0_i8mm(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out) {
    const auto* weights = static_cast<const BlockQ4_0*>(w);
    const auto* activations = static_cast<const BlockQ8_0*>(x);
    const std::size_t blocks = cols / kBlockSize;

    std::size_t r = 0;
    for (; r + 4 <= rows; r += 4) {
        std::size_t t = 0;
        for (; t + 4 <= n; t += 4) {
            tile_4x4(weights + r * blocks, blocks, activations + t * blocks, blocks, out + t * rows + r, rows);
        }
        // Leftover tokens: one row at a time with the dot-product kernel.
        for (; t < n; ++t) {
            for (std::size_t i = 0; i < 4; ++i) {
                out[t * rows + r + i] = neon::dot_row(weights + (r + i) * blocks, activations + t * blocks, blocks, dot_block);
            }
        }
    }
    // Leftover rows.
    for (; r < rows; ++r) {
        for (std::size_t t = 0; t < n; ++t) {
            out[t * rows + r] = neon::dot_row(weights + r * blocks, activations + t * blocks, blocks, dot_block);
        }
    }
}

namespace {

// A pair of tokens' block, pre-arranged for smmla: four chunk vectors of
// [token a 8 elements | token b 8 elements], the pair's scales as
// [da, db, da, db], and 8 * (da * sum qa, db * sum qb, ...) for removing the
// Q4_0 offset of 8 after an unsigned multiply.
struct TokenPairBlock {
    int8x16_t b[4];
    float32x4_t scales;
    float32x4_t offset;
};

// The prepared activations: one TokenPairBlock per token pair per block,
// followed by the raw Q8_0 rows (for the single-token tail).
struct Prepared {
    const TokenPairBlock* pairs;
    const BlockQ8_0* rows;
};

inline Prepared view_prepared(const void* x, std::size_t n, std::size_t blocks) {
    const auto* pairs = static_cast<const TokenPairBlock*>(x);
    return {pairs, reinterpret_cast<const BlockQ8_0*>(pairs + (n / 2) * blocks)};
}

}  // namespace

std::size_t prepared_bytes_q4_0x4_i8mm(std::size_t n, std::size_t blocks) {
    return (n / 2) * blocks * sizeof(TokenPairBlock) + n * blocks * sizeof(BlockQ8_0);
}

void prepare_q4_0x4_i8mm(const void* x_raw, std::size_t n, std::size_t blocks, void* out_raw) {
    const auto* x = static_cast<const BlockQ8_0*>(x_raw);
    auto* prepared = static_cast<TokenPairBlock*>(out_raw);
    const std::size_t pairs = n / 2;
    std::memcpy(prepared + pairs * blocks, x, n * blocks * sizeof(BlockQ8_0));
    for (std::size_t p = 0; p < pairs; ++p) {
        const BlockQ8_0* ta = x + (2 * p) * blocks;
        const BlockQ8_0* tb = ta + blocks;
        for (std::size_t blk = 0; blk < blocks; ++blk) {
            TokenPairBlock& out = prepared[p * blocks + blk];
            zip_tokens(ta[blk], tb[blk], out.b);
            out.scales = token_scales(ta[blk].d, tb[blk].d);
            const float sum_a = static_cast<float>(vaddlvq_s8(vld1q_s8(ta[blk].qs)) + vaddlvq_s8(vld1q_s8(ta[blk].qs + 16)));
            const float sum_b = static_cast<float>(vaddlvq_s8(vld1q_s8(tb[blk].qs)) + vaddlvq_s8(vld1q_s8(tb[blk].qs + 16)));
            out.offset = vmulq_n_f32(vmulq_f32(out.scales, vcombine_f32(vset_lane_f32(sum_b, vdup_n_f32(sum_a), 1),
                                                                        vset_lane_f32(sum_b, vdup_n_f32(sum_a), 1))),
                                     8.0f);
        }
    }
}

namespace {

inline int32x4_t usmmla4(const uint8x16_t a[4], const int8x16_t b[4]) {
    int32x4_t acc = vusmmlaq_s32(vdupq_n_s32(0), a[0], b[0]);
    acc = vusmmlaq_s32(acc, a[1], b[1]);
    acc = vusmmlaq_s32(acc, a[2], b[2]);
    return vusmmlaq_s32(acc, a[3], b[3]);
}

// 4 packed rows x (2 * kPairs) tokens. Weights stay as unsigned nibbles (0..15);
// the offset of 8 is removed afterwards: sum((q - 8) x) = sum(q x) - 8 sum(x).
template <int kPairs>
void tile_x4(const BlockQ4_0x4* group, std::size_t blocks, const TokenPairBlock* pair_blocks, float* out,
             std::size_t out_stride) {
    const uint8x16_t low_mask = vdupq_n_u8(0x0F);
    float32x4_t acc[2][kPairs];
    for (int rp = 0; rp < 2; ++rp) {
        for (int tp = 0; tp < kPairs; ++tp) acc[rp][tp] = vdupq_n_f32(0.0f);
    }
    for (std::size_t blk = 0; blk < blocks; ++blk) {
        const BlockQ4_0x4& wb = group[blk];
        uint8x16_t a[2][4];
        for (int rp = 0; rp < 2; ++rp) {
            const uint8x16_t v0 = vld1q_u8(wb.qs + 32 * rp);
            const uint8x16_t v1 = vld1q_u8(wb.qs + 32 * rp + 16);
            a[rp][0] = vandq_u8(v0, low_mask);  // chunk 0
            a[rp][1] = vandq_u8(v1, low_mask);  // chunk 1
            a[rp][2] = vshrq_n_u8(v0, 4);       // chunk 2
            a[rp][3] = vshrq_n_u8(v1, 4);       // chunk 3
        }
        const float32x4_t d = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(wb.d)));
        const float32x4_t rs[2] = {vzip1q_f32(d, d), vzip2q_f32(d, d)};  // [d0 d0 d1 d1], [d2 d2 d3 d3]
        for (int tp = 0; tp < kPairs; ++tp) {
            const TokenPairBlock& pair = pair_blocks[static_cast<std::size_t>(tp) * blocks + blk];
            for (int rp = 0; rp < 2; ++rp) {
                const float32x4_t scaled = vmulq_f32(rs[rp], pair.scales);
                acc[rp][tp] = vmlaq_f32(acc[rp][tp], vcvtq_f32_s32(usmmla4(a[rp], pair.b)), scaled);
                acc[rp][tp] = vmlsq_f32(acc[rp][tp], rs[rp], pair.offset);
            }
        }
    }
    for (std::size_t rp = 0; rp < 2; ++rp) {
        for (std::size_t tp = 0; tp < kPairs; ++tp) {
            float* o = out + 2 * tp * out_stride + 2 * rp;
            o[0] = vgetq_lane_f32(acc[rp][tp], 0);
            o[out_stride] = vgetq_lane_f32(acc[rp][tp], 1);
            o[1] = vgetq_lane_f32(acc[rp][tp], 2);
            o[out_stride + 1] = vgetq_lane_f32(acc[rp][tp], 3);
        }
    }
}

}  // namespace

void matmul_q4_0x4_i8mm(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out) {
    const auto* weights = static_cast<const BlockQ4_0x4*>(w);
    const std::size_t blocks = cols / kBlockSize;
    const Prepared prepared = view_prepared(x, n, blocks);
    const TokenPairBlock* pairs = prepared.pairs;
    const BlockQ8_0* activations = prepared.rows;
    for (std::size_t g = 0; g < rows / 4; ++g) {
        std::size_t t = 0;
        for (; t + 8 <= n; t += 8) {
            tile_x4<4>(weights + g * blocks, blocks, pairs + (t / 2) * blocks, out + t * rows + 4 * g, rows);
        }
        for (; t + 4 <= n; t += 4) {
            tile_x4<2>(weights + g * blocks, blocks, pairs + (t / 2) * blocks, out + t * rows + 4 * g, rows);
        }
        for (; t < n; ++t) {
            matvec_q4_0x4_dotprod(weights + g * blocks, 4, cols, activations + t * blocks, out + t * rows + 4 * g);
        }
    }
}

// ---- 256-element formats: TQ2_0x4, Q4_Kx4, Q6_Kx4 ------------------------------

namespace {

// A pair of tokens' 256-element block for the smmla tiles: 32 chunk vectors
// of [token a 8 elements | token b 8 elements], the pair's scales
// [da db da db], and each token's sums per 16 elements.
struct KTokenPairBlock {
    int8x16_t b[32];
    float32x4_t scales;
    int16x8_t bsums[2][2];  // [token][0: sub-blocks 0-7, 1: 8-15]
};

struct KPrepared {
    const KTokenPairBlock* pairs;
    const BlockQ8_K* rows;
};

inline KPrepared view_k_prepared(const void* x, std::size_t n, std::size_t blocks) {
    const auto* pairs = static_cast<const KTokenPairBlock*>(x);
    return {pairs, reinterpret_cast<const BlockQ8_K*>(pairs + (n / 2) * blocks)};
}

template <int kShift>
inline uint8x16_t ternary_layer_u8(uint8x16_t packed) {
    if constexpr (kShift == 0) return vandq_u8(packed, vdupq_n_u8(3));
    else if constexpr (kShift == 6) return vshrq_n_u8(packed, 6);
    else return vandq_u8(vshrq_n_u8(packed, kShift), vdupq_n_u8(3));
}

// [sum over token a, sum over token b, same, same] of a pair's 16-element sums.
inline int32x4_t pair_totals(const KTokenPairBlock& pair) {
    const std::int32_t a = vaddlvq_s16(vaddq_s16(pair.bsums[0][0], pair.bsums[0][1]));
    const std::int32_t b = vaddlvq_s16(vaddq_s16(pair.bsums[1][0], pair.bsums[1][1]));
    return int32x4_t{a, b, a, b};
}

// Stores the 2x2 tile results held in accf[rp][tp] lanes (r0t0, r0t1, r1t0, r1t1).
template <int kPairs>
inline void store_tile(const float32x4_t accf[2][kPairs], float* out, std::size_t out_stride) {
    for (std::size_t rp = 0; rp < 2; ++rp) {
        for (std::size_t tp = 0; tp < kPairs; ++tp) {
            float* o = out + 2 * tp * out_stride + 2 * rp;
            o[0] = vgetq_lane_f32(accf[rp][tp], 0);
            o[out_stride] = vgetq_lane_f32(accf[rp][tp], 1);
            o[1] = vgetq_lane_f32(accf[rp][tp], 2);
            o[out_stride + 1] = vgetq_lane_f32(accf[rp][tp], 3);
        }
    }
}

// 4 packed ternary rows x (2 * kPairs) tokens over a whole row of blocks.
// Weights stay 0..2; the offset of 1 is removed per block: sum((q-1) x) = sum(q x) - sum(x).
template <int kPairs>
void ternary_tile(const BlockTQ2_0x4* group, std::size_t blocks, const KTokenPairBlock* pair_blocks, float* out,
                  std::size_t out_stride) {
    float32x4_t accf[2][kPairs];
    for (int rp = 0; rp < 2; ++rp) {
        for (int tp = 0; tp < kPairs; ++tp) accf[rp][tp] = vdupq_n_f32(0.0f);
    }
    for (std::size_t blk = 0; blk < blocks; ++blk) {
        const BlockTQ2_0x4& wb = group[blk];
        int32x4_t acc[2][kPairs];
        for (int rp = 0; rp < 2; ++rp) {
            for (int tp = 0; tp < kPairs; ++tp) acc[rp][tp] = vdupq_n_s32(0);
        }
        for (int v = 0; v < 8; ++v) {
            const uint8x16_t w0 = vld1q_u8(wb.qs + 16 * v);
            const uint8x16_t w1 = vld1q_u8(wb.qs + 128 + 16 * v);
            const uint8x16_t a[2][4] = {
                {ternary_layer_u8<0>(w0), ternary_layer_u8<2>(w0), ternary_layer_u8<4>(w0), ternary_layer_u8<6>(w0)},
                {ternary_layer_u8<0>(w1), ternary_layer_u8<2>(w1), ternary_layer_u8<4>(w1), ternary_layer_u8<6>(w1)}};
            for (int tp = 0; tp < kPairs; ++tp) {
                const int8x16_t* b = pair_blocks[static_cast<std::size_t>(tp) * blocks + blk].b + 4 * v;
                for (int l = 0; l < 4; ++l) {
                    acc[0][tp] = vusmmlaq_s32(acc[0][tp], a[0][l], b[l]);
                    acc[1][tp] = vusmmlaq_s32(acc[1][tp], a[1][l], b[l]);
                }
            }
        }
        const float32x4_t d = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(wb.d)));
        const float32x4_t rs[2] = {vzip1q_f32(d, d), vzip2q_f32(d, d)};
        for (int tp = 0; tp < kPairs; ++tp) {
            const KTokenPairBlock& pair = pair_blocks[static_cast<std::size_t>(tp) * blocks + blk];
            const int32x4_t totals = pair_totals(pair);
            for (int rp = 0; rp < 2; ++rp) {
                const int32x4_t corrected = vsubq_s32(acc[rp][tp], totals);
                accf[rp][tp] = vmlaq_f32(accf[rp][tp], vcvtq_f32_s32(corrected), vmulq_f32(rs[rp], pair.scales));
            }
        }
    }
    store_tile<kPairs>(accf, out, out_stride);
}

// 4 packed Q4_K rows x (2 * kPairs) tokens. Sub-block scales are applied in
// the integer domain; the per-block float step then needs only d and dmin.
template <int kPairs>
void q4_k_tile(const BlockQ4_Kx4* group, std::size_t blocks, const KTokenPairBlock* pair_blocks, float* out,
               std::size_t out_stride) {
    const uint8x16_t low_mask = vdupq_n_u8(0x0F);
    float32x4_t accf[2][kPairs];
    for (int rp = 0; rp < 2; ++rp) {
        for (int tp = 0; tp < kPairs; ++tp) accf[rp][tp] = vdupq_n_f32(0.0f);
    }
    for (std::size_t blk = 0; blk < blocks; ++blk) {
        const BlockQ4_Kx4& wb = group[blk];
        std::int16_t scales[4][8], mins[4][8];
        for (int r = 0; r < 4; ++r) {
            for (int sb = 0; sb < 8; ++sb) {
                std::uint8_t sc, mn;
                q4_k_scale_min(wb.scales[r], sb, sc, mn);
                scales[r][sb] = sc;
                mins[r][sb] = mn;
            }
        }
        int32x4_t acc[2][kPairs];
        for (int rp = 0; rp < 2; ++rp) {
            for (int tp = 0; tp < kPairs; ++tp) acc[rp][tp] = vdupq_n_s32(0);
        }
        for (int sb = 0; sb < 8; ++sb) {
            uint8x16_t a[2][4];
            int32x4_t scale[2];
            for (int rp = 0; rp < 2; ++rp) {
                const uint8x16_t v0 = vld1q_u8(wb.qs + 256 * rp + 32 * sb);
                const uint8x16_t v1 = vld1q_u8(wb.qs + 256 * rp + 32 * sb + 16);
                a[rp][0] = vandq_u8(v0, low_mask);
                a[rp][1] = vandq_u8(v1, low_mask);
                a[rp][2] = vshrq_n_u8(v0, 4);
                a[rp][3] = vshrq_n_u8(v1, 4);
                scale[rp] = int32x4_t{scales[2 * rp][sb], scales[2 * rp][sb], scales[2 * rp + 1][sb], scales[2 * rp + 1][sb]};
            }
            for (int tp = 0; tp < kPairs; ++tp) {
                const int8x16_t* b = pair_blocks[static_cast<std::size_t>(tp) * blocks + blk].b + 4 * sb;
                for (int rp = 0; rp < 2; ++rp) {
                    int32x4_t sum = vusmmlaq_s32(vdupq_n_s32(0), a[rp][0], b[0]);
                    sum = vusmmlaq_s32(sum, a[rp][1], b[1]);
                    sum = vusmmlaq_s32(sum, a[rp][2], b[2]);
                    sum = vusmmlaq_s32(sum, a[rp][3], b[3]);
                    acc[rp][tp] = vmlaq_s32(acc[rp][tp], sum, scale[rp]);
                }
            }
        }
        const float32x4_t d = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(wb.d)));
        const float32x4_t dmin = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(wb.dmin)));
        const float32x4_t rs[2] = {vzip1q_f32(d, d), vzip2q_f32(d, d)};
        const float32x4_t rmin[2] = {vzip1q_f32(dmin, dmin), vzip2q_f32(dmin, dmin)};
        for (int tp = 0; tp < kPairs; ++tp) {
            const KTokenPairBlock& pair = pair_blocks[static_cast<std::size_t>(tp) * blocks + blk];
            // Sums per sub-block of 32 for each token, as int16x8.
            const int16x8_t bsum32[2] = {vpaddq_s16(pair.bsums[0][0], pair.bsums[0][1]),
                                         vpaddq_s16(pair.bsums[1][0], pair.bsums[1][1])};
            for (int rp = 0; rp < 2; ++rp) {
                // offsets lanes (r0 ta, r0 tb, r1 ta, r1 tb) = sum over sub-blocks of min * bsum32
                const int16x8_t m0 = vld1q_s16(mins[2 * rp]);
                const int16x8_t m1 = vld1q_s16(mins[2 * rp + 1]);
                auto dot = [](int16x8_t p, int16x8_t q) {
                    return vaddvq_s32(vmlal_s16(vmull_s16(vget_low_s16(p), vget_low_s16(q)), vget_high_s16(p), vget_high_s16(q)));
                };
                const int32x4_t offsets = {dot(m0, bsum32[0]), dot(m0, bsum32[1]), dot(m1, bsum32[0]), dot(m1, bsum32[1])};
                accf[rp][tp] = vmlaq_f32(accf[rp][tp], vcvtq_f32_s32(acc[rp][tp]), vmulq_f32(rs[rp], pair.scales));
                accf[rp][tp] = vmlsq_f32(accf[rp][tp], vcvtq_f32_s32(offsets), vmulq_f32(rmin[rp], pair.scales));
            }
        }
    }
    store_tile<kPairs>(accf, out, out_stride);
}

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

// 4 packed Q6_K rows x (2 * kPairs) tokens. Values are 0..63 with the offset
// of 32 still in; it is removed per block via the activation sums.
template <int kPairs>
void q6_k_tile(const BlockQ6_Kx4* group, std::size_t blocks, const KTokenPairBlock* pair_blocks, float* out,
               std::size_t out_stride) {
    float32x4_t accf[2][kPairs];
    for (int rp = 0; rp < 2; ++rp) {
        for (int tp = 0; tp < kPairs; ++tp) accf[rp][tp] = vdupq_n_f32(0.0f);
    }
    for (std::size_t blk = 0; blk < blocks; ++blk) {
        const BlockQ6_Kx4& wb = group[blk];
        int32x4_t acc[2][kPairs];
        for (int rp = 0; rp < 2; ++rp) {
            for (int tp = 0; tp < kPairs; ++tp) acc[rp][tp] = vdupq_n_s32(0);
        }
        for (int grp = 0; grp < 4; ++grp) {
            uint8x16_t a[2][8];
            unpack_q6_kx4_group(wb.low[0][grp], wb.high[0][grp], a[0]);
            unpack_q6_kx4_group(wb.low[1][grp], wb.high[1][grp], a[1]);
            for (int j = 0; j < 4; ++j) {
                const int sb = 4 * grp + j;
                const int32x4_t scale[2] = {
                    int32x4_t{wb.scales[0][sb], wb.scales[0][sb], wb.scales[1][sb], wb.scales[1][sb]},
                    int32x4_t{wb.scales[2][sb], wb.scales[2][sb], wb.scales[3][sb], wb.scales[3][sb]}};
                for (int tp = 0; tp < kPairs; ++tp) {
                    const int8x16_t* b = pair_blocks[static_cast<std::size_t>(tp) * blocks + blk].b + 8 * grp + 2 * j;
                    for (int rp = 0; rp < 2; ++rp) {
                        int32x4_t sum = vusmmlaq_s32(vdupq_n_s32(0), a[rp][2 * j], b[0]);
                        sum = vusmmlaq_s32(sum, a[rp][2 * j + 1], b[1]);
                        acc[rp][tp] = vmlaq_s32(acc[rp][tp], sum, scale[rp]);
                    }
                }
            }
        }
        const float32x4_t d = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(wb.d)));
        const float32x4_t rs[2] = {vzip1q_f32(d, d), vzip2q_f32(d, d)};
        for (int tp = 0; tp < kPairs; ++tp) {
            const KTokenPairBlock& pair = pair_blocks[static_cast<std::size_t>(tp) * blocks + blk];
            for (int rp = 0; rp < 2; ++rp) {
                // correction lanes = 32 * sum over sub-blocks of scale * bsum16, per (row, token)
                auto dot = [&](int r, int t) {
                    const int16x8_t s0 = vmovl_s8(vld1_s8(wb.scales[r]));
                    const int16x8_t s1 = vmovl_s8(vld1_s8(wb.scales[r] + 8));
                    int32x4_t v = vmull_s16(vget_low_s16(s0), vget_low_s16(pair.bsums[t][0]));
                    v = vmlal_s16(v, vget_high_s16(s0), vget_high_s16(pair.bsums[t][0]));
                    v = vmlal_s16(v, vget_low_s16(s1), vget_low_s16(pair.bsums[t][1]));
                    v = vmlal_s16(v, vget_high_s16(s1), vget_high_s16(pair.bsums[t][1]));
                    return vaddvq_s32(v);
                };
                const int32x4_t correction = {dot(2 * rp, 0), dot(2 * rp, 1), dot(2 * rp + 1, 0), dot(2 * rp + 1, 1)};
                const int32x4_t corrected = vsubq_s32(acc[rp][tp], vshlq_n_s32(correction, 5));
                accf[rp][tp] = vmlaq_f32(accf[rp][tp], vcvtq_f32_s32(corrected), vmulq_f32(rs[rp], pair.scales));
            }
        }
    }
    store_tile<kPairs>(accf, out, out_stride);
}

template <typename Block, typename Tile8, typename Tile4, typename Matvec>
void grouped_k_matmul(const Block* weights, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out,
                      Tile8 tile8, Tile4 tile4, Matvec matvec) {
    const std::size_t blocks = cols / kSuperBlockSize;
    const KPrepared prepared = view_k_prepared(x, n, blocks);
    for (std::size_t g = 0; g < rows / 4; ++g) {
        std::size_t t = 0;
        for (; t + 8 <= n; t += 8) tile8(weights + g * blocks, blocks, prepared.pairs + (t / 2) * blocks, out + t * rows + 4 * g, rows);
        for (; t + 4 <= n; t += 4) tile4(weights + g * blocks, blocks, prepared.pairs + (t / 2) * blocks, out + t * rows + 4 * g, rows);
        for (; t < n; ++t) matvec(weights + g * blocks, 4, cols, prepared.rows + t * blocks, out + t * rows + 4 * g);
    }
}

}  // namespace

std::size_t prepared_bytes_k_i8mm(std::size_t n, std::size_t blocks) {
    return (n / 2) * blocks * sizeof(KTokenPairBlock) + n * blocks * sizeof(BlockQ8_K);
}

void prepare_k_i8mm(const void* x_raw, std::size_t n, std::size_t blocks, void* out_raw) {
    const auto* x = static_cast<const BlockQ8_K*>(x_raw);
    auto* prepared = static_cast<KTokenPairBlock*>(out_raw);
    const std::size_t pairs = n / 2;
    std::memcpy(prepared + pairs * blocks, x, n * blocks * sizeof(BlockQ8_K));
    for (std::size_t p = 0; p < pairs; ++p) {
        const BlockQ8_K* ta = x + (2 * p) * blocks;
        const BlockQ8_K* tb = ta + blocks;
        for (std::size_t blk = 0; blk < blocks; ++blk) {
            KTokenPairBlock& out = prepared[p * blocks + blk];
            for (int c = 0; c < 32; ++c) out.b[c] = vcombine_s8(vld1_s8(ta[blk].qs + 8 * c), vld1_s8(tb[blk].qs + 8 * c));
            const float32x2_t d = vset_lane_f32(tb[blk].d, vdup_n_f32(ta[blk].d), 1);
            out.scales = vcombine_f32(d, d);
            out.bsums[0][0] = vld1q_s16(ta[blk].bsums);
            out.bsums[0][1] = vld1q_s16(ta[blk].bsums + 8);
            out.bsums[1][0] = vld1q_s16(tb[blk].bsums);
            out.bsums[1][1] = vld1q_s16(tb[blk].bsums + 8);
        }
    }
}

void matmul_tq2_0x4_i8mm(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out) {
    grouped_k_matmul(static_cast<const BlockTQ2_0x4*>(w), rows, cols, x, n, out, ternary_tile<4>, ternary_tile<2>,
                     matvec_tq2_0x4_dotprod);
}

void matmul_q4_kx4_i8mm(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out) {
    grouped_k_matmul(static_cast<const BlockQ4_Kx4*>(w), rows, cols, x, n, out, q4_k_tile<4>, q4_k_tile<2>,
                     matvec_q4_kx4_dotprod);
}

void matmul_q6_kx4_i8mm(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out) {
    grouped_k_matmul(static_cast<const BlockQ6_Kx4*>(w), rows, cols, x, n, out, q6_k_tile<4>, q6_k_tile<2>,
                     matvec_q6_kx4_dotprod);
}

}  // namespace brisk::kernels
