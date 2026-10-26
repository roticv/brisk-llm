// Baseline NEON (ARMv8.0) kernels: these run on the Raspberry Pi 4's
// Cortex-A72, which has no 8-bit dot-product instruction, so the products are
// widened to 16 bits and summed pairwise.

#include <vector>

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

// For 8-bit weights the 16-bit lanes would overflow (4 * 127 * 127), so each
// 16-product partial is widened to 32 bits before the next is added.
inline int32x4_t dot_block_wide(int8x16_t w_low, int8x16_t w_high, const BlockQ8_0& x) {
    const int8x16_t x_low = vld1q_s8(x.qs);
    const int8x16_t x_high = vld1q_s8(x.qs + 16);
    int32x4_t acc = vpaddlq_s16(vmull_s8(vget_low_s8(w_low), vget_low_s8(x_low)));
    acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(w_low), vget_high_s8(x_low)));
    acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(w_high), vget_low_s8(x_high)));
    acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(w_high), vget_high_s8(x_high)));
    return acc;
}

inline int32x4_t dot16(int8x16_t w, int8x16_t x) {
    const int16x8_t p = vmlal_s8(vmull_s8(vget_low_s8(w), vget_low_s8(x)), vget_high_s8(w), vget_high_s8(x));
    return vpaddlq_s16(p);
}

const auto unpack_q5 = [](const BlockQ5_0& b, int8x16_t& l, int8x16_t& h) { neon::unpack_q5_0(b, l, h); };
const auto unpack_q8 = [](const BlockQ8_0& b, int8x16_t& l, int8x16_t& h) { neon::unpack_q8_0(b, l, h); };

}  // namespace

void matvec_q4_0_neon(const void* w, std::size_t rows, std::size_t cols, const void* x, float* out) {
    const auto* weights = static_cast<const BlockQ4_0*>(w);
    const auto* activations = static_cast<const BlockQ8_0*>(x);
    const std::size_t blocks = cols / kBlockSize;
    for (std::size_t r = 0; r < rows; ++r) out[r] = neon::dot_row(weights + r * blocks, activations, blocks, dot_block);
}

void matvec_q4_1_neon(const void* w, std::size_t rows, std::size_t cols, const void* x, float* out) {
    const auto* weights = static_cast<const BlockQ4_1*>(w);
    const auto* activations = static_cast<const BlockQ8_0*>(x);
    const std::size_t blocks = cols / kBlockSize;
    for (std::size_t r = 0; r < rows; ++r) out[r] = neon::dot_row_q4_1(weights + r * blocks, activations, blocks, dot_block);
}

void matmul_q4_1_neon(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out,
                      std::size_t out_stride) {
    const auto* weights = static_cast<const BlockQ4_1*>(w);
    const auto* activations = static_cast<const BlockQ8_0*>(x);
    const std::size_t blocks = cols / kBlockSize;
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t t = 0; t < n; ++t) {
            out[t * out_stride + r] = neon::dot_row_q4_1(weights + r * blocks, activations + t * blocks, blocks, dot_block);
        }
    }
}

void matvec_q5_0_neon(const void* w, std::size_t rows, std::size_t cols, const void* x, float* out) {
    const auto* weights = static_cast<const BlockQ5_0*>(w);
    const auto* activations = static_cast<const BlockQ8_0*>(x);
    const std::size_t blocks = cols / kBlockSize;
    for (std::size_t r = 0; r < rows; ++r) {
        out[r] = neon::dot_row_any(weights + r * blocks, activations, blocks, unpack_q5, dot_block);
    }
}

void matmul_q5_0_neon(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out,
                      std::size_t out_stride) {
    neon::matmul_any(static_cast<const BlockQ5_0*>(w), rows, cols, static_cast<const BlockQ8_0*>(x), n, out, out_stride,
                     unpack_q5, dot_block);
}

void matvec_q8_0_neon(const void* w, std::size_t rows, std::size_t cols, const void* x, float* out) {
    const auto* weights = static_cast<const BlockQ8_0*>(w);
    const auto* activations = static_cast<const BlockQ8_0*>(x);
    const std::size_t blocks = cols / kBlockSize;
    for (std::size_t r = 0; r < rows; ++r) {
        out[r] = neon::dot_row_any(weights + r * blocks, activations, blocks, unpack_q8, dot_block_wide);
    }
}

void matmul_q8_0_neon(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out,
                      std::size_t out_stride) {
    neon::matmul_any(static_cast<const BlockQ8_0*>(w), rows, cols, static_cast<const BlockQ8_0*>(x), n, out, out_stride,
                     unpack_q8, dot_block_wide);
}

void matmul_q4_0_neon(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out,
                      std::size_t out_stride) {
    neon::matmul(static_cast<const BlockQ4_0*>(w), rows, cols, static_cast<const BlockQ8_0*>(x), n, out, out_stride, dot_block);
}

void matvec_q5_k_neon(const void* w, std::size_t rows, std::size_t cols, const void* x, float* out) {
    neon::matmul_k(static_cast<const BlockQ5_K*>(w), rows, cols, static_cast<const BlockQ8_K*>(x), 1, out, rows,
                   [](const BlockQ5_K* row, const BlockQ8_K* const* xs, std::size_t n, std::size_t blocks, float* o) {
                       neon::dot_rows_k4_wide(row, xs, n, blocks, o, [](const BlockQ5_K& b, int8x16_t* q, std::uint8_t* sc, std::uint8_t* mn) { neon::unpack_q5_k(b, q, sc, mn); });
                   });
}

void matmul_q5_k_neon(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out,
                      std::size_t out_stride) {
    neon::matmul_k<neon::kWideTokens>(static_cast<const BlockQ5_K*>(w), rows, cols, static_cast<const BlockQ8_K*>(x), n, out, out_stride,
                   [](const BlockQ5_K* row, const BlockQ8_K* const* xs, std::size_t count, std::size_t blocks, float* o) {
                       neon::dot_rows_k4_wide(row, xs, count, blocks, o, [](const BlockQ5_K& b, int8x16_t* q, std::uint8_t* sc, std::uint8_t* mn) { neon::unpack_q5_k(b, q, sc, mn); });
                   });
}

void matvec_q4_k_neon(const void* w, std::size_t rows, std::size_t cols, const void* x, float* out) {
    neon::matmul_k(static_cast<const BlockQ4_K*>(w), rows, cols, static_cast<const BlockQ8_K*>(x), 1, out, rows,
                   [](const BlockQ4_K* row, const BlockQ8_K* const* xs, std::size_t n, std::size_t blocks, float* o) {
                       neon::dot_rows_k4_wide(row, xs, n, blocks, o, [](const BlockQ4_K& b, int8x16_t* q, std::uint8_t* sc, std::uint8_t* mn) { neon::unpack_q4_k(b, q, sc, mn); });
                   });
}

void matmul_q4_k_neon(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out,
                      std::size_t out_stride) {
    neon::matmul_k<neon::kWideTokens>(static_cast<const BlockQ4_K*>(w), rows, cols, static_cast<const BlockQ8_K*>(x), n, out, out_stride,
                   [](const BlockQ4_K* row, const BlockQ8_K* const* xs, std::size_t count, std::size_t blocks, float* o) {
                       neon::dot_rows_k4_wide(row, xs, count, blocks, o, [](const BlockQ4_K& b, int8x16_t* q, std::uint8_t* sc, std::uint8_t* mn) { neon::unpack_q4_k(b, q, sc, mn); });
                   });
}

void matvec_q6_k_neon(const void* w, std::size_t rows, std::size_t cols, const void* x, float* out) {
    neon::matmul_k(static_cast<const BlockQ6_K*>(w), rows, cols, static_cast<const BlockQ8_K*>(x), 1, out, rows,
                   [](const BlockQ6_K* row, const BlockQ8_K* const* xs, std::size_t n, std::size_t blocks, float* o) {
                       neon::dot_rows_q6_k(row, xs, n, blocks, o, dot16);
                   });
}

void matmul_q6_k_neon(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out,
                      std::size_t out_stride) {
    neon::matmul_k(static_cast<const BlockQ6_K*>(w), rows, cols, static_cast<const BlockQ8_K*>(x), n, out, out_stride,
                   [](const BlockQ6_K* row, const BlockQ8_K* const* xs, std::size_t count, std::size_t blocks, float* o) {
                       neon::dot_rows_q6_k(row, xs, count, blocks, o, dot16);
                   });
}

void matvec_tq2_0_neon(const void* w, std::size_t rows, std::size_t cols, const void* x, float* out) {
    neon::matmul_k(static_cast<const BlockTQ2_0*>(w), rows, cols, static_cast<const BlockQ8_K*>(x), 1, out, rows,
                   [](const BlockTQ2_0* row, const BlockQ8_K* const* xs, std::size_t n, std::size_t blocks, float* o) {
                       neon::dot_rows_tq2_0_wide(row, xs, n, blocks, o);
                   });
}

void matmul_tq2_0_neon(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out,
                      std::size_t out_stride) {
    neon::matmul_k<neon::kTernaryTokens>(static_cast<const BlockTQ2_0*>(w), rows, cols, static_cast<const BlockQ8_K*>(x), n, out, out_stride,
                   [](const BlockTQ2_0* row, const BlockQ8_K* const* xs, std::size_t count, std::size_t blocks, float* o) {
                       neon::dot_rows_tq2_0_wide(row, xs, count, blocks, o);
                   });
}

// ---- the lookup-table ternary kernel -------------------------------------
//
// bitnet.cpp's idea: for each column pair of the activation row, a 16-byte
// table of the nine sums w0 x0 + w1 x1 (w in -1, 0, 1), indexed by the pair's
// ternary code w0 * 3 + w1. A row group's 16 indices for a pair are one
// `tbl` away from 16 rows' partial sums, so the multiplier is never used; the
// bytes are summed in 16-bit lanes per 256-column block, then scaled. The
// activations are 7-bit so a pair's sum fits a byte (activation_bits).
namespace {

// The tables for one activation block: 128 pairs x 16 bytes.
inline void ternary_tables(const BlockQ8_K& x, std::int8_t* tables) {
    for (std::size_t p = 0; p < 128; ++p) {
        const std::int8_t x0 = x.qs[2 * p], x1 = x.qs[2 * p + 1];
        std::int8_t* t = tables + 16 * p;
        t[0] = static_cast<std::int8_t>(-x0 - x1);
        t[1] = static_cast<std::int8_t>(-x0);
        t[2] = static_cast<std::int8_t>(-x0 + x1);
        t[3] = static_cast<std::int8_t>(-x1);
        t[4] = 0;
        t[5] = x1;
        t[6] = static_cast<std::int8_t>(x0 - x1);
        t[7] = x0;
        t[8] = static_cast<std::int8_t>(x0 + x1);
        for (int i = 9; i < 16; ++i) t[i] = 0;
    }
}

// 16 rows of one block against one token's tables: the 16 sums as int32.
inline void ternary_lookup_block(const std::uint8_t* qs, const std::int8_t* tables, int32x4_t acc[4]) {
    const uint8x16_t low_mask = vdupq_n_u8(0x0F);
    int16x8_t lo = vdupq_n_s16(0), hi = vdupq_n_s16(0);
    for (std::size_t k = 0; k < 64; ++k) {  // chunk k: pairs 2k (low nibbles) and 2k + 1 (high)
        const uint8x16_t packed = vld1q_u8(qs + 16 * k);
        const int8x16_t t0 = vld1q_s8(tables + 32 * k), t1 = vld1q_s8(tables + 32 * k + 16);
        const int8x16_t v0 = vqtbl1q_s8(t0, vandq_u8(packed, low_mask));
        const int8x16_t v1 = vqtbl1q_s8(t1, vshrq_n_u8(packed, 4));
        lo = vaddw_s8(lo, vget_low_s8(v0));
        hi = vaddw_s8(hi, vget_high_s8(v0));
        lo = vaddw_s8(lo, vget_low_s8(v1));
        hi = vaddw_s8(hi, vget_high_s8(v1));
    }
    acc[0] = vmovl_s16(vget_low_s16(lo));
    acc[1] = vmovl_s16(vget_high_s16(lo));
    acc[2] = vmovl_s16(vget_low_s16(hi));
    acc[3] = vmovl_s16(vget_high_s16(hi));
}

// The prepared form of one token's activation block: its scale, then the tables.
struct PreparedBlock {
    float d;
    float pad[3];
    std::int8_t tables[2048];
};
static_assert(sizeof(PreparedBlock) == 2064);

}  // namespace

std::size_t prepared_bytes_tq2_0x16(std::size_t n, std::size_t blocks) { return n * blocks * sizeof(PreparedBlock); }

void prepare_tq2_0x16(const void* x, std::size_t n, std::size_t blocks, void* out) {
    const auto* activations = static_cast<const BlockQ8_K*>(x);
    auto* prepared = static_cast<PreparedBlock*>(out);
    for (std::size_t i = 0; i < n * blocks; ++i) {
        prepared[i].d = activations[i].d;
        ternary_tables(activations[i], prepared[i].tables);
    }
}

void matmul_tq2_0x16_neon(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out,
                          std::size_t out_stride) {
    const auto* weights = static_cast<const BlockTQ2_0x16*>(w);
    const auto* prepared = static_cast<const PreparedBlock*>(x);
    const std::size_t blocks = cols / kSuperBlockSize;
    // Per row group, the blocks outermost and the tile's tokens inner: one 1 KB
    // weight block and the tile's tables for it (2 KB per token) stay in L1
    // while the tile's sums wait in a small array. (Tables are eight times the
    // activations they stand for, so the working set, not the instruction count,
    // decides this kernel's speed on the A72.)
    constexpr std::size_t kTile = 8;
    for (std::size_t t0 = 0; t0 < n; t0 += kTile) {
        const std::size_t t1 = std::min(n, t0 + kTile);
        for (std::size_t g = 0; g < rows / 16; ++g) {
            const BlockTQ2_0x16* group = weights + g * blocks;
            float32x4_t sums[kTile][4];
            for (std::size_t t = 0; t < kTile; ++t) {
                for (int k = 0; k < 4; ++k) sums[t][k] = vdupq_n_f32(0.0f);
            }
            for (std::size_t b = 0; b < blocks; ++b) {
                float32x4_t wd[4];
                for (int k = 0; k < 4; ++k) wd[k] = neon::half4_to_float4(group[b].d + 4 * k);
                for (std::size_t t = t0; t < t1; ++t) {
                    const PreparedBlock& p = prepared[t * blocks + b];
                    int32x4_t acc[4];
                    ternary_lookup_block(group[b].qs, p.tables, acc);
                    const float32x4_t xd = vdupq_n_f32(p.d);
                    for (int k = 0; k < 4; ++k) {
                        sums[t - t0][k] = vmlaq_f32(sums[t - t0][k], vcvtq_f32_s32(acc[k]), vmulq_f32(xd, wd[k]));
                    }
                }
            }
            for (std::size_t t = t0; t < t1; ++t) {
                float* o = out + t * out_stride + 16 * g;
                for (int k = 0; k < 4; ++k) vst1q_f32(o + 4 * k, sums[t - t0][k]);
            }
        }
    }
}

void matvec_tq2_0x16_neon(const void* w, std::size_t rows, std::size_t cols, const void* x, float* out) {
    // One token: blocks outermost so the block's 2 KB table stays in L1 while the
    // row groups' weights for that block stream past; the row sums live in `out`.
    const auto* weights = static_cast<const BlockTQ2_0x16*>(w);
    const auto* prepared = static_cast<const PreparedBlock*>(x);
    const std::size_t blocks = cols / kSuperBlockSize;
    const std::size_t groups = rows / 16;
    for (std::size_t g = 0; g < groups; ++g) {
        for (int k = 0; k < 4; ++k) vst1q_f32(out + 16 * g + 4 * k, vdupq_n_f32(0.0f));
    }
    for (std::size_t b = 0; b < blocks; ++b) {
        const PreparedBlock& p = prepared[b];
        const float32x4_t xd = vdupq_n_f32(p.d);
        for (std::size_t g = 0; g < groups; ++g) {
            const BlockTQ2_0x16& block = weights[g * blocks + b];
            int32x4_t acc[4];
            ternary_lookup_block(block.qs, p.tables, acc);
            float* o = out + 16 * g;
            for (int k = 0; k < 4; ++k) {
                const float32x4_t scale = vmulq_f32(xd, neon::half4_to_float4(block.d + 4 * k));
                vst1q_f32(o + 4 * k, vmlaq_f32(vld1q_f32(o + 4 * k), vcvtq_f32_s32(acc[k]), scale));
            }
        }
    }
}

}  // namespace brisk::kernels
