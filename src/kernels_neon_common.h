#pragma once

// Shared by the NEON kernel variants. Each variant supplies the 8-bit block
// dot product; everything around it (nibble unpacking, scales, accumulation)
// is the same.

#include <arm_neon.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "quant.h"

namespace brisk::kernels::neon {

inline void unpack_q4_0(const BlockQ4_0& block, int8x16_t& low, int8x16_t& high) {
    const uint8x16_t packed = vld1q_u8(block.qs);
    const int8x16_t eight = vdupq_n_s8(8);
    low = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(packed, vdupq_n_u8(0x0F))), eight);
    high = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(packed, 4)), eight);
}

// Product of the four blocks' weight and activation scales, converted from
// half precision in hardware.
inline float32x4_t scales4(const BlockQ4_0* w, const BlockQ8_0* x) {
    uint16x4_t wd = vdup_n_u16(w[0].d);
    wd = vset_lane_u16(w[1].d, wd, 1);
    wd = vset_lane_u16(w[2].d, wd, 2);
    wd = vset_lane_u16(w[3].d, wd, 3);
    uint16x4_t xd = vdup_n_u16(x[0].d);
    xd = vset_lane_u16(x[1].d, xd, 1);
    xd = vset_lane_u16(x[2].d, xd, 2);
    xd = vset_lane_u16(x[3].d, xd, 3);
    return vmulq_f32(vcvt_f32_f16(vreinterpret_f16_u16(wd)), vcvt_f32_f16(vreinterpret_f16_u16(xd)));
}

inline float scale1(const BlockQ4_0& w, const BlockQ8_0& x) {
    return vgetq_lane_f32(vcvt_f32_f16(vreinterpret_f16_u16(vdup_n_u16(w.d))), 0) *
           vgetq_lane_f32(vcvt_f32_f16(vreinterpret_f16_u16(vdup_n_u16(x.d))), 0);
}

// DotBlock(w_low, w_high, x) returns the 32 products of one block as four
// 32-bit partial sums.
template <typename DotBlock>
inline float dot_row(const BlockQ4_0* w, const BlockQ8_0* x, std::size_t blocks, DotBlock dot_block) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    std::size_t b = 0;
    for (; b + 4 <= blocks; b += 4) {
        int8x16_t l0, h0, l1, h1, l2, h2, l3, h3;
        unpack_q4_0(w[b], l0, h0);
        unpack_q4_0(w[b + 1], l1, h1);
        unpack_q4_0(w[b + 2], l2, h2);
        unpack_q4_0(w[b + 3], l3, h3);
        const int32x4_t s0 = dot_block(l0, h0, x[b]);
        const int32x4_t s1 = dot_block(l1, h1, x[b + 1]);
        const int32x4_t s2 = dot_block(l2, h2, x[b + 2]);
        const int32x4_t s3 = dot_block(l3, h3, x[b + 3]);
        // One lane per block: [sum s0, sum s1, sum s2, sum s3]
        const int32x4_t sums = vpaddq_s32(vpaddq_s32(s0, s1), vpaddq_s32(s2, s3));
        acc = vmlaq_f32(acc, vcvtq_f32_s32(sums), scales4(w + b, x + b));
    }
    float total = vaddvq_f32(acc);
    for (; b < blocks; ++b) {
        int8x16_t l, h;
        unpack_q4_0(w[b], l, h);
        total += static_cast<float>(vaddvq_s32(dot_block(l, h, x[b]))) * scale1(w[b], x[b]);
    }
    return total;
}

// One weight row against four activation rows. The weights are unpacked once
// per block and reused, which is what makes prompt processing compute-bound.
template <typename DotBlock>
inline void dot_row_x4(const BlockQ4_0* w, const BlockQ8_0* const x[4], std::size_t blocks, float out[4],
                       DotBlock dot_block) {
    float32x4_t acc[4] = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
    std::size_t b = 0;
    for (; b + 4 <= blocks; b += 4) {
        int8x16_t l0, h0, l1, h1, l2, h2, l3, h3;
        unpack_q4_0(w[b], l0, h0);
        unpack_q4_0(w[b + 1], l1, h1);
        unpack_q4_0(w[b + 2], l2, h2);
        unpack_q4_0(w[b + 3], l3, h3);
        uint16x4_t wd = vdup_n_u16(w[b].d);
        wd = vset_lane_u16(w[b + 1].d, wd, 1);
        wd = vset_lane_u16(w[b + 2].d, wd, 2);
        wd = vset_lane_u16(w[b + 3].d, wd, 3);
        const float32x4_t w_scales = vcvt_f32_f16(vreinterpret_f16_u16(wd));
        for (int t = 0; t < 4; ++t) {
            const BlockQ8_0* xt = x[t] + b;
            const int32x4_t s0 = dot_block(l0, h0, xt[0]);
            const int32x4_t s1 = dot_block(l1, h1, xt[1]);
            const int32x4_t s2 = dot_block(l2, h2, xt[2]);
            const int32x4_t s3 = dot_block(l3, h3, xt[3]);
            const int32x4_t sums = vpaddq_s32(vpaddq_s32(s0, s1), vpaddq_s32(s2, s3));
            uint16x4_t xd = vdup_n_u16(xt[0].d);
            xd = vset_lane_u16(xt[1].d, xd, 1);
            xd = vset_lane_u16(xt[2].d, xd, 2);
            xd = vset_lane_u16(xt[3].d, xd, 3);
            const float32x4_t scales = vmulq_f32(w_scales, vcvt_f32_f16(vreinterpret_f16_u16(xd)));
            acc[t] = vmlaq_f32(acc[t], vcvtq_f32_s32(sums), scales);
        }
    }
    for (int t = 0; t < 4; ++t) out[t] = vaddvq_f32(acc[t]);
    for (; b < blocks; ++b) {
        int8x16_t l, h;
        unpack_q4_0(w[b], l, h);
        for (int t = 0; t < 4; ++t) {
            out[t] += static_cast<float>(vaddvq_s32(dot_block(l, h, x[t][b]))) * scale1(w[b], x[t][b]);
        }
    }
}

template <typename DotBlock>
inline void matmul(const BlockQ4_0* w, std::size_t rows, std::size_t cols, const BlockQ8_0* x, std::size_t n, float* out,
                   DotBlock dot_block) {
    const std::size_t blocks = cols / kBlockSize;
    for (std::size_t r = 0; r < rows; ++r) {
        const BlockQ4_0* row = w + r * blocks;
        std::size_t t = 0;
        for (; t + 4 <= n; t += 4) {
            const BlockQ8_0* const xs[4] = {x + t * blocks, x + (t + 1) * blocks, x + (t + 2) * blocks,
                                            x + (t + 3) * blocks};
            float result[4];
            dot_row_x4(row, xs, blocks, result, dot_block);
            for (int i = 0; i < 4; ++i) out[(t + static_cast<std::size_t>(i)) * rows + r] = result[i];
        }
        for (; t < n; ++t) out[t * rows + r] = dot_row(row, x + t * blocks, blocks, dot_block);
    }
}

// ---- K-quants -------------------------------------------------------------
//
// Dot16(w, x) returns the 16 products of two 8-bit vectors as four 32-bit
// partial sums; the variants differ only in how they compute it.

inline float32x4_t half4_to_float(std::uint16_t h) {
    return vcvt_f32_f16(vreinterpret_f16_u16(vdup_n_u16(h)));
}

// Decodes a Q6_K block into 16 vectors of 16 signed values, offset removed,
// in weight order (so vector g holds the elements scaled by scales[g]).
inline void unpack_q6_k(const BlockQ6_K& block, int8x16_t q[16]) {
    const int8x16_t offset = vdupq_n_s8(32);
    const uint8x16_t low_mask = vdupq_n_u8(0x0F);
    for (int half = 0; half < 2; ++half) {
        const std::uint8_t* ql = block.ql + half * 64;
        const std::uint8_t* qh = block.qh + half * 32;
        int8x16_t* out = q + half * 8;
        for (int v = 0; v < 2; ++v) {  // elements l = 16v .. 16v+15 of this half
            const uint8x16_t l0 = vld1q_u8(ql + 16 * v);
            const uint8x16_t l1 = vld1q_u8(ql + 32 + 16 * v);
            const uint8x16_t h = vld1q_u8(qh + 16 * v);
            const uint8x16_t q1 = vorrq_u8(vandq_u8(l0, low_mask), vshlq_n_u8(vandq_u8(h, vdupq_n_u8(3)), 4));
            const uint8x16_t q2 = vorrq_u8(vandq_u8(l1, low_mask), vshlq_n_u8(vandq_u8(vshrq_n_u8(h, 2), vdupq_n_u8(3)), 4));
            const uint8x16_t q3 = vorrq_u8(vshrq_n_u8(l0, 4), vshlq_n_u8(vandq_u8(vshrq_n_u8(h, 4), vdupq_n_u8(3)), 4));
            const uint8x16_t q4 = vorrq_u8(vshrq_n_u8(l1, 4), vshlq_n_u8(vshrq_n_u8(h, 6), 4));
            out[0 + v] = vsubq_s8(vreinterpretq_s8_u8(q1), offset);  // elements l
            out[2 + v] = vsubq_s8(vreinterpretq_s8_u8(q2), offset);  // elements l + 32
            out[4 + v] = vsubq_s8(vreinterpretq_s8_u8(q3), offset);  // elements l + 64
            out[6 + v] = vsubq_s8(vreinterpretq_s8_u8(q4), offset);  // elements l + 96
        }
    }
}

// Decodes a Q4_K block into 16 vectors of 16 unsigned nibbles in weight order
// (vectors 2j and 2j+1 are sub-block j) and its 8 scales and minimums.
inline void unpack_q4_k(const BlockQ4_K& block, int8x16_t q[16], std::uint8_t scales[8], std::uint8_t mins[8]) {
    const uint8x16_t low_mask = vdupq_n_u8(0x0F);
    for (int chunk = 0; chunk < 4; ++chunk) {
        const uint8x16_t c0 = vld1q_u8(block.qs + chunk * 32);
        const uint8x16_t c1 = vld1q_u8(block.qs + chunk * 32 + 16);
        q[chunk * 4 + 0] = vreinterpretq_s8_u8(vandq_u8(c0, low_mask));
        q[chunk * 4 + 1] = vreinterpretq_s8_u8(vandq_u8(c1, low_mask));
        q[chunk * 4 + 2] = vreinterpretq_s8_u8(vshrq_n_u8(c0, 4));
        q[chunk * 4 + 3] = vreinterpretq_s8_u8(vshrq_n_u8(c1, 4));
    }
    for (int j = 0; j < 8; ++j) q4_k_scale_min(block.scales, j, scales[j], mins[j]);
}

// Sum over one Q6_K block for one activation block, before the block scales.
template <typename Dot16>
inline std::int32_t q6_k_block_sum(const int8x16_t q[16], const std::int8_t* scales, const BlockQ8_K& x, Dot16 dot16) {
    int32x4_t acc0 = vdupq_n_s32(0);
    int32x4_t acc1 = vdupq_n_s32(0);
    for (int g = 0; g < 16; g += 2) {
        acc0 = vmlaq_n_s32(acc0, dot16(q[g], vld1q_s8(x.qs + 16 * g)), scales[g]);
        acc1 = vmlaq_n_s32(acc1, dot16(q[g + 1], vld1q_s8(x.qs + 16 * g + 16)), scales[g + 1]);
    }
    return vaddvq_s32(vaddq_s32(acc0, acc1));
}

template <typename Dot16>
inline float q4_k_block_value(const int8x16_t q[16], const std::uint8_t scales[8], const std::uint8_t mins[8],
                              const BlockQ4_K& w, const BlockQ8_K& x, Dot16 dot16) {
    int32x4_t acc0 = vdupq_n_s32(0);
    int32x4_t acc1 = vdupq_n_s32(0);
    std::int32_t offsets = 0;
    for (int j = 0; j < 8; ++j) {
        acc0 = vmlaq_n_s32(acc0, dot16(q[2 * j], vld1q_s8(x.qs + 32 * j)), scales[j]);
        acc1 = vmlaq_n_s32(acc1, dot16(q[2 * j + 1], vld1q_s8(x.qs + 32 * j + 16)), scales[j]);
        offsets += mins[j] * (x.bsums[2 * j] + x.bsums[2 * j + 1]);
    }
    const float weighted = static_cast<float>(vaddvq_s32(vaddq_s32(acc0, acc1)));
    return x.d * (vgetq_lane_f32(half4_to_float(w.d), 0) * weighted -
                  vgetq_lane_f32(half4_to_float(w.dmin), 0) * static_cast<float>(offsets));
}

// Weight row against up to 4 activation rows; each block is unpacked once.
template <typename Dot16>
inline void dot_rows_q6_k(const BlockQ6_K* w, const BlockQ8_K* const* xs, std::size_t n, std::size_t blocks,
                          float* out, Dot16 dot16) {
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (std::size_t b = 0; b < blocks; ++b) {
        int8x16_t q[16];
        unpack_q6_k(w[b], q);
        const float d = vgetq_lane_f32(half4_to_float(w[b].d), 0);
        for (std::size_t t = 0; t < n; ++t) {
            sums[t] += xs[t][b].d * d * static_cast<float>(q6_k_block_sum(q, w[b].scales, xs[t][b], dot16));
        }
    }
    for (std::size_t t = 0; t < n; ++t) out[t] = sums[t];
}

template <typename Dot16>
inline void dot_rows_q4_k(const BlockQ4_K* w, const BlockQ8_K* const* xs, std::size_t n, std::size_t blocks,
                          float* out, Dot16 dot16) {
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (std::size_t b = 0; b < blocks; ++b) {
        int8x16_t q[16];
        std::uint8_t scales[8], mins[8];
        unpack_q4_k(w[b], q, scales, mins);
        for (std::size_t t = 0; t < n; ++t) sums[t] += q4_k_block_value(q, scales, mins, w[b], xs[t][b], dot16);
    }
    for (std::size_t t = 0; t < n; ++t) out[t] = sums[t];
}

// Generic driver for the K-quant kernels: `dot_rows(w_row, xs, n, blocks, out)`.
template <typename Block, typename DotRows>
inline void matmul_k(const Block* w, std::size_t rows, std::size_t cols, const BlockQ8_K* x, std::size_t n, float* out,
                     DotRows dot_rows) {
    const std::size_t blocks = cols / kSuperBlockSize;
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t t = 0; t < n; t += 4) {
            const std::size_t count = std::min<std::size_t>(4, n - t);
            const BlockQ8_K* xs[4] = {x + t * blocks, x + (t + 1) * blocks, x + (t + 2) * blocks, x + (t + 3) * blocks};
            float result[4];
            dot_rows(w + r * blocks, xs, count, blocks, result);
            for (std::size_t i = 0; i < count; ++i) out[(t + i) * rows + r] = result[i];
        }
    }
}

}  // namespace brisk::kernels::neon
