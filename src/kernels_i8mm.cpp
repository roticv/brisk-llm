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

}  // namespace brisk::kernels
