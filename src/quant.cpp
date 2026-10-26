#include "quant.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace brisk {

std::size_t block_elements(WeightFormat format) {
    switch (format) {
        case WeightFormat::F32: return 1;
        case WeightFormat::Q4_0:
        case WeightFormat::Q4_0x4:
        case WeightFormat::Q4_1: return kBlockSize;
        case WeightFormat::Q5_0: return kBlockSize;
        case WeightFormat::Q5_0x4: return kBlockSize;
        case WeightFormat::Q8_0: return kBlockSize;
        case WeightFormat::Q4_K:
        case WeightFormat::Q5_K:
        case WeightFormat::Q6_K:
        case WeightFormat::TQ2_0:
        case WeightFormat::TQ2_0x4:
        case WeightFormat::TQ2_0x16:
        case WeightFormat::Q4_Kx4:
        case WeightFormat::Q6_Kx4: return kSuperBlockSize;
    }
    return 1;
}

std::size_t block_bytes(WeightFormat format) {
    switch (format) {
        case WeightFormat::F32: return sizeof(float);
        case WeightFormat::Q4_0: return sizeof(BlockQ4_0);
        case WeightFormat::Q4_0x4: return sizeof(BlockQ4_0x4) / 4;  // per row
        case WeightFormat::Q4_1: return sizeof(BlockQ4_1);
        case WeightFormat::Q5_0: return sizeof(BlockQ5_0);
        case WeightFormat::Q5_0x4: return sizeof(BlockQ5_0x4) / 4;  // per row
        case WeightFormat::Q8_0: return sizeof(BlockQ8_0);
        case WeightFormat::Q4_K: return sizeof(BlockQ4_K);
        case WeightFormat::Q5_K: return sizeof(BlockQ5_K);
        case WeightFormat::Q6_K: return sizeof(BlockQ6_K);
        case WeightFormat::TQ2_0: return sizeof(BlockTQ2_0);
        case WeightFormat::TQ2_0x4: return sizeof(BlockTQ2_0x4) / 4;  // per row
        case WeightFormat::TQ2_0x16: return sizeof(BlockTQ2_0x16) / 16;
        case WeightFormat::Q4_Kx4: return sizeof(BlockQ4_Kx4) / 4;
        case WeightFormat::Q6_Kx4: return sizeof(BlockQ6_Kx4) / 4;
    }
    return 0;
}

bool takes_q8_k(WeightFormat format) {
    return format == WeightFormat::Q4_K || format == WeightFormat::Q5_K || format == WeightFormat::Q6_K || format == WeightFormat::TQ2_0 ||
           format == WeightFormat::TQ2_0x4 || format == WeightFormat::TQ2_0x16 || format == WeightFormat::Q4_Kx4 ||
           format == WeightFormat::Q6_Kx4;
}

float half_to_float(std::uint16_t half) {
    const std::uint32_t h = half;
    const std::uint32_t sign = (h & 0x8000) << 16;
    std::uint32_t exponent = (h >> 10) & 0x1F;
    std::uint32_t mantissa = h & 0x3FF;
    std::uint32_t bits = 0;
    if (exponent == 0x1F) {
        bits = sign | 0x7F800000 | (mantissa << 13);  // infinity or NaN
    } else if (exponent != 0) {
        bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
    } else if (mantissa != 0) {
        // Subnormal half: shift the mantissa up until its leading bit is implicit.
        exponent = 113;
        while ((mantissa & 0x400) == 0) {
            mantissa <<= 1;
            --exponent;
        }
        bits = sign | (exponent << 23) | ((mantissa & 0x3FF) << 13);
    } else {
        bits = sign;
    }
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

std::uint16_t float_to_half(float f) {
    std::uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & 0x8000;
    const std::int32_t exponent = static_cast<std::int32_t>((bits >> 23) & 0xFF) - 127 + 15;
    std::uint32_t mantissa = bits & 0x7FFFFF;

    if (((bits >> 23) & 0xFF) == 0xFF) {  // infinity or NaN
        return static_cast<std::uint16_t>(sign | 0x7C00 | (mantissa != 0 ? 0x200 : 0));
    }
    if (exponent >= 0x1F) return static_cast<std::uint16_t>(sign | 0x7C00);  // overflow to infinity
    if (exponent <= 0) {
        if (exponent < -10) return static_cast<std::uint16_t>(sign);  // underflow to zero
        // Subnormal half: include the implicit bit and shift right, rounding to nearest even.
        mantissa |= 0x800000;
        const std::uint32_t shift = static_cast<std::uint32_t>(14 - exponent);
        std::uint32_t half = mantissa >> shift;
        const std::uint32_t remainder = mantissa & ((1u << shift) - 1);
        const std::uint32_t halfway = 1u << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (half & 1))) ++half;
        return static_cast<std::uint16_t>(sign | half);
    }
    std::uint32_t half = (static_cast<std::uint32_t>(exponent) << 10) | (mantissa >> 13);
    const std::uint32_t remainder = mantissa & 0x1FFF;
    if (remainder > 0x1000 || (remainder == 0x1000 && (half & 1))) ++half;  // may carry into the exponent, correctly
    return static_cast<std::uint16_t>(sign | half);
}

void quantize_q8_0(const float* x, BlockQ8_0* out, std::size_t count) {
    for (std::size_t b = 0; b < count / kBlockSize; ++b) {
        const float* block = x + b * kBlockSize;
        float amax = 0.0f;
        for (std::size_t i = 0; i < kBlockSize; ++i) amax = std::max(amax, std::fabs(block[i]));
        // As llama.cpp does: the scale is stored as half, but the values are
        // computed from the unrounded scale, with round-to-nearest-even.
        const float d = amax / 127.0f;
        const float inverse = d != 0.0f ? 1.0f / d : 0.0f;
        out[b].d = float_to_half(d);
#if defined(__aarch64__)
        const float32x4_t vinv = vdupq_n_f32(inverse);
        for (std::size_t i = 0; i < kBlockSize; i += 16) {
            const int32x4_t v0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(block + i), vinv));
            const int32x4_t v1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(block + i + 4), vinv));
            const int32x4_t v2 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(block + i + 8), vinv));
            const int32x4_t v3 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(block + i + 12), vinv));
            const int16x8_t lo = vcombine_s16(vmovn_s32(v0), vmovn_s32(v1));
            const int16x8_t hi = vcombine_s16(vmovn_s32(v2), vmovn_s32(v3));
            vst1q_s8(out[b].qs + i, vcombine_s8(vmovn_s16(lo), vmovn_s16(hi)));
        }
#else
        for (std::size_t i = 0; i < kBlockSize; ++i) {
            out[b].qs[i] = static_cast<std::int8_t>(std::nearbyint(block[i] * inverse));
        }
#endif
    }
}

int activation_bits(WeightFormat format) { return format == WeightFormat::TQ2_0x16 ? 7 : 8; }

void quantize_q8_k(const float* x, BlockQ8_K* out, std::size_t count, int bits) {
    for (std::size_t b = 0; b < count / kSuperBlockSize; ++b) {
        const float* block = x + b * kSuperBlockSize;
        BlockQ8_K& y = out[b];
        // As ggml does: scale by the signed extreme value, so it maps to -127.
        float max = 0.0f;
        float amax = 0.0f;
        for (std::size_t i = 0; i < kSuperBlockSize; ++i) {
            const float ax = std::fabs(block[i]);
            if (ax > amax) {
                amax = ax;
                max = block[i];
            }
        }
        if (amax == 0.0f) {
            y.d = 0.0f;
            std::memset(y.qs, 0, sizeof(y.qs));
            std::memset(y.bsums, 0, sizeof(y.bsums));
            continue;
        }
        const float extreme = bits == 7 ? 63.0f : 127.0f;
        const float inverse = -extreme / max;
        y.d = 1.0f / inverse;
#if defined(__aarch64__)
        const float32x4_t vinv = vdupq_n_f32(inverse);
        const int32x4_t cap = vdupq_n_s32(static_cast<std::int32_t>(extreme));
        for (std::size_t g = 0; g < kSuperBlockSize / 16; ++g) {
            const float* p = block + g * 16;
            const int32x4_t v0 = vminq_s32(vcvtnq_s32_f32(vmulq_f32(vld1q_f32(p), vinv)), cap);
            const int32x4_t v1 = vminq_s32(vcvtnq_s32_f32(vmulq_f32(vld1q_f32(p + 4), vinv)), cap);
            const int32x4_t v2 = vminq_s32(vcvtnq_s32_f32(vmulq_f32(vld1q_f32(p + 8), vinv)), cap);
            const int32x4_t v3 = vminq_s32(vcvtnq_s32_f32(vmulq_f32(vld1q_f32(p + 12), vinv)), cap);
            const int16x8_t lo = vcombine_s16(vmovn_s32(v0), vmovn_s32(v1));
            const int16x8_t hi = vcombine_s16(vmovn_s32(v2), vmovn_s32(v3));
            vst1q_s8(y.qs + g * 16, vcombine_s8(vmovn_s16(lo), vmovn_s16(hi)));
            y.bsums[g] = static_cast<std::int16_t>(vaddvq_s16(vaddq_s16(lo, hi)));
        }
#else
        for (std::size_t i = 0; i < kSuperBlockSize; ++i) {
            y.qs[i] = static_cast<std::int8_t>(std::min(127.0f, std::nearbyint(inverse * block[i])));
        }
        for (std::size_t g = 0; g < kSuperBlockSize / 16; ++g) {
            int sum = 0;
            for (std::size_t i = 0; i < 16; ++i) sum += y.qs[g * 16 + i];
            y.bsums[g] = static_cast<std::int16_t>(sum);
        }
#endif
    }
}

void dequantize_q4_0(const BlockQ4_0* blocks, float* out, std::size_t count) {
    for (std::size_t b = 0; b < count / kBlockSize; ++b) {
        const float d = half_to_float(blocks[b].d);
        float* y = out + b * kBlockSize;
        for (std::size_t i = 0; i < kBlockSize / 2; ++i) {
            y[i] = static_cast<float>((blocks[b].qs[i] & 0x0F) - 8) * d;
            y[i + kBlockSize / 2] = static_cast<float>((blocks[b].qs[i] >> 4) - 8) * d;
        }
    }
}

void dequantize_q4_1(const BlockQ4_1* blocks, float* out, std::size_t count) {
    for (std::size_t b = 0; b < count / kBlockSize; ++b) {
        const float d = half_to_float(blocks[b].d);
        const float m = half_to_float(blocks[b].m);
        float* y = out + b * kBlockSize;
        for (std::size_t i = 0; i < kBlockSize / 2; ++i) {
            y[i] = static_cast<float>(blocks[b].qs[i] & 0x0F) * d + m;
            y[i + kBlockSize / 2] = static_cast<float>(blocks[b].qs[i] >> 4) * d + m;
        }
    }
}

void dequantize_q5_0(const BlockQ5_0* blocks, float* out, std::size_t count) {
    for (std::size_t b = 0; b < count / kBlockSize; ++b) {
        const float d = half_to_float(blocks[b].d);
        std::uint32_t qh;
        std::memcpy(&qh, blocks[b].qh, 4);
        float* y = out + b * kBlockSize;
        for (std::size_t i = 0; i < kBlockSize / 2; ++i) {
            const int lo = (blocks[b].qs[i] & 0x0F) | static_cast<int>(((qh >> i) & 1) << 4);
            const int hi = (blocks[b].qs[i] >> 4) | static_cast<int>(((qh >> (i + 16)) & 1) << 4);
            y[i] = static_cast<float>(lo - 16) * d;
            y[i + kBlockSize / 2] = static_cast<float>(hi - 16) * d;
        }
    }
}

void dequantize_q8_0(const BlockQ8_0* blocks, float* out, std::size_t count) {
    for (std::size_t b = 0; b < count / kBlockSize; ++b) {
        const float d = half_to_float(blocks[b].d);
        for (std::size_t i = 0; i < kBlockSize; ++i) out[b * kBlockSize + i] = static_cast<float>(blocks[b].qs[i]) * d;
    }
}

void dequantize_q4_k(const BlockQ4_K* blocks, float* out, std::size_t count) {
    for (std::size_t b = 0; b < count / kSuperBlockSize; ++b) {
        const BlockQ4_K& block = blocks[b];
        const float d = half_to_float(block.d);
        const float dmin = half_to_float(block.dmin);
        float* y = out + b * kSuperBlockSize;
        // Each 64-weight chunk holds the low nibbles of one sub-block of 32, then the high nibbles of the next.
        for (int chunk = 0; chunk < 4; ++chunk) {
            std::uint8_t scale, min;
            q4_k_scale_min(block.scales, 2 * chunk, scale, min);
            const float d1 = d * scale, m1 = dmin * min;
            q4_k_scale_min(block.scales, 2 * chunk + 1, scale, min);
            const float d2 = d * scale, m2 = dmin * min;
            const std::uint8_t* q = block.qs + chunk * 32;
            for (int i = 0; i < 32; ++i) {
                y[chunk * 64 + i] = d1 * static_cast<float>(q[i] & 0xF) - m1;
                y[chunk * 64 + 32 + i] = d2 * static_cast<float>(q[i] >> 4) - m2;
            }
        }
    }
}

void dequantize_q5_k(const BlockQ5_K* blocks, float* out, std::size_t count) {
    for (std::size_t b = 0; b < count / kSuperBlockSize; ++b) {
        const BlockQ5_K& block = blocks[b];
        const float d = half_to_float(block.d);
        const float dmin = half_to_float(block.dmin);
        float* y = out + b * kSuperBlockSize;
        for (int chunk = 0; chunk < 4; ++chunk) {
            std::uint8_t scale, min;
            q4_k_scale_min(block.scales, 2 * chunk, scale, min);
            const float d1 = d * scale, m1 = dmin * min;
            q4_k_scale_min(block.scales, 2 * chunk + 1, scale, min);
            const float d2 = d * scale, m2 = dmin * min;
            const std::uint8_t* q = block.qs + chunk * 32;
            const std::uint8_t lo_bit = static_cast<std::uint8_t>(1u << (2 * chunk)), hi_bit = static_cast<std::uint8_t>(lo_bit << 1);
            for (int i = 0; i < 32; ++i) {
                y[chunk * 64 + i] = d1 * static_cast<float>((q[i] & 0xF) | ((block.qh[i] & lo_bit) ? 16 : 0)) - m1;
                y[chunk * 64 + 32 + i] = d2 * static_cast<float>((q[i] >> 4) | ((block.qh[i] & hi_bit) ? 16 : 0)) - m2;
            }
        }
    }
}

void unpack_q6_k(const BlockQ6_K& block, std::int8_t* q) {
    for (int half = 0; half < 2; ++half) {
        const std::uint8_t* ql = block.ql + half * 64;
        const std::uint8_t* qh = block.qh + half * 32;
        std::int8_t* y = q + half * 128;
        for (int l = 0; l < 32; ++l) {
            y[l] = static_cast<std::int8_t>(((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
            y[l + 32] = static_cast<std::int8_t>(((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
            y[l + 64] = static_cast<std::int8_t>(((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
            y[l + 96] = static_cast<std::int8_t>(((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
        }
    }
}

void dequantize_q6_k(const BlockQ6_K* blocks, float* out, std::size_t count) {
    std::int8_t q[kSuperBlockSize];
    for (std::size_t b = 0; b < count / kSuperBlockSize; ++b) {
        const float d = half_to_float(blocks[b].d);
        unpack_q6_k(blocks[b], q);
        float* y = out + b * kSuperBlockSize;
        for (std::size_t i = 0; i < kSuperBlockSize; ++i) y[i] = d * blocks[b].scales[i / 16] * q[i];
    }
}

namespace {

// Where element e (0..31) of row r (0..3) of a Q4_0x4 block lives: byte index
// and whether it is the high nibble.
inline void q4_0x4_position(std::size_t r, std::size_t e, std::size_t& byte, bool& high) {
    const std::size_t chunk = e / 8;  // 0..3: elements 0-7, 8-15, 16-23, 24-31
    high = chunk >= 2;
    const std::size_t k = (r / 2) * 2 + (chunk % 2);  // which 16-byte vector
    byte = k * 16 + (r % 2) * 8 + (e % 8);
}

}  // namespace

void repack_q4_0x4(const BlockQ4_0* in, std::size_t rows, std::size_t blocks, BlockQ4_0x4* out) {
    for (std::size_t g = 0; g < rows / 4; ++g) {
        for (std::size_t b = 0; b < blocks; ++b) {
            BlockQ4_0x4& packed = out[g * blocks + b];
            std::memset(packed.qs, 0, sizeof(packed.qs));
            for (std::size_t r = 0; r < 4; ++r) {
                const BlockQ4_0& block = in[(g * 4 + r) * blocks + b];
                packed.d[r] = block.d;
                for (std::size_t e = 0; e < kBlockSize; ++e) {
                    const std::uint8_t q = e < 16 ? (block.qs[e] & 0x0F) : (block.qs[e - 16] >> 4);
                    std::size_t byte;
                    bool high;
                    q4_0x4_position(r, e, byte, high);
                    packed.qs[byte] |= static_cast<std::uint8_t>(high ? q << 4 : q);
                }
            }
        }
    }
}

void dequantize_q4_0x4_row(const BlockQ4_0x4* group, std::size_t row_in_group, float* out, std::size_t count) {
    for (std::size_t b = 0; b < count / kBlockSize; ++b) {
        const float d = half_to_float(group[b].d[row_in_group]);
        for (std::size_t e = 0; e < kBlockSize; ++e) {
            std::size_t byte;
            bool high;
            q4_0x4_position(row_in_group, e, byte, high);
            const int q = high ? (group[b].qs[byte] >> 4) : (group[b].qs[byte] & 0x0F);
            out[b * kBlockSize + e] = static_cast<float>(q - 8) * d;
        }
    }
}

// Where the fifth bit of element e of row r lives in a Q5_0x4 block: byte of qh and bit.
inline void q5_0x4_bit_position(std::size_t r, std::size_t e, std::size_t& byte, int& bit) {
    std::size_t nibble_byte;
    bool high;
    q4_0x4_position(r, e, nibble_byte, high);
    const std::size_t k = nibble_byte / 16, lane = nibble_byte % 16;
    byte = 4 * k + (high ? 2 : 0) + lane / 8;
    bit = static_cast<int>(lane % 8);
}

void repack_q5_0x4(const BlockQ5_0* in, std::size_t rows, std::size_t blocks, BlockQ5_0x4* out) {
    for (std::size_t g = 0; g < rows / 4; ++g) {
        for (std::size_t b = 0; b < blocks; ++b) {
            BlockQ5_0x4& packed = out[g * blocks + b];
            std::memset(packed.qs, 0, sizeof(packed.qs));
            std::memset(packed.qh, 0, sizeof(packed.qh));
            for (std::size_t r = 0; r < 4; ++r) {
                const BlockQ5_0& block = in[(g * 4 + r) * blocks + b];
                packed.d[r] = block.d;
                std::uint32_t qh;
                std::memcpy(&qh, block.qh, 4);
                for (std::size_t e = 0; e < kBlockSize; ++e) {
                    const std::uint8_t q = e < 16 ? (block.qs[e] & 0x0F) : (block.qs[e - 16] >> 4);
                    std::size_t byte;
                    bool high;
                    q4_0x4_position(r, e, byte, high);
                    packed.qs[byte] |= static_cast<std::uint8_t>(high ? q << 4 : q);
                    std::size_t hbyte;
                    int bit;
                    q5_0x4_bit_position(r, e, hbyte, bit);
                    packed.qh[hbyte] |= static_cast<std::uint8_t>(((qh >> e) & 1) << bit);
                }
            }
        }
    }
}

void dequantize_q5_0x4_row(const BlockQ5_0x4* group, std::size_t row_in_group, float* out, std::size_t count) {
    for (std::size_t b = 0; b < count / kBlockSize; ++b) {
        const float d = half_to_float(group[b].d[row_in_group]);
        for (std::size_t e = 0; e < kBlockSize; ++e) {
            std::size_t byte, hbyte;
            bool high;
            int bit;
            q4_0x4_position(row_in_group, e, byte, high);
            q5_0x4_bit_position(row_in_group, e, hbyte, bit);
            const int q = (high ? (group[b].qs[byte] >> 4) : (group[b].qs[byte] & 0x0F)) | (((group[b].qh[hbyte] >> bit) & 1) << 4);
            out[b * kBlockSize + e] = static_cast<float>(q - 16) * d;
        }
    }
}

void dot_q5_0x4_q8_0(const BlockQ5_0x4* w, const BlockQ8_0* x, std::size_t blocks, float out[4]) {
    for (std::size_t r = 0; r < 4; ++r) {
        float sum = 0.0f;
        for (std::size_t b = 0; b < blocks; ++b) {
            std::int32_t acc = 0;
            for (std::size_t e = 0; e < kBlockSize; ++e) {
                std::size_t byte, hbyte;
                bool high;
                int bit;
                q4_0x4_position(r, e, byte, high);
                q5_0x4_bit_position(r, e, hbyte, bit);
                const int q = (high ? (w[b].qs[byte] >> 4) : (w[b].qs[byte] & 0x0F)) | (((w[b].qh[hbyte] >> bit) & 1) << 4);
                acc += (q - 16) * x[b].qs[e];
            }
            sum += static_cast<float>(acc) * half_to_float(w[b].d[r]) * half_to_float(x[b].d);
        }
        out[r] = sum;
    }
}

void dot_q4_0x4_q8_0(const BlockQ4_0x4* w, const BlockQ8_0* x, std::size_t blocks, float out[4]) {
    for (std::size_t r = 0; r < 4; ++r) {
        float sum = 0.0f;
        for (std::size_t b = 0; b < blocks; ++b) {
            std::int32_t acc = 0;
            for (std::size_t e = 0; e < kBlockSize; ++e) {
                std::size_t byte;
                bool high;
                q4_0x4_position(r, e, byte, high);
                const int q = high ? (w[b].qs[byte] >> 4) : (w[b].qs[byte] & 0x0F);
                acc += (q - 8) * x[b].qs[e];
            }
            sum += static_cast<float>(acc) * half_to_float(w[b].d[r]) * half_to_float(x[b].d);
        }
        out[r] = sum;
    }
}

float dot_q4_0_q8_0(const BlockQ4_0* w, const BlockQ8_0* x, std::size_t blocks) {
    float sum = 0.0f;
    for (std::size_t b = 0; b < blocks; ++b) {
        std::int32_t acc = 0;
        for (std::size_t i = 0; i < kBlockSize / 2; ++i) {
            acc += ((w[b].qs[i] & 0x0F) - 8) * x[b].qs[i];
            acc += ((w[b].qs[i] >> 4) - 8) * x[b].qs[i + kBlockSize / 2];
        }
        sum += static_cast<float>(acc) * half_to_float(w[b].d) * half_to_float(x[b].d);
    }
    return sum;
}

float dot_q5_0_q8_0(const BlockQ5_0* w, const BlockQ8_0* x, std::size_t blocks) {
    float sum = 0.0f;
    for (std::size_t b = 0; b < blocks; ++b) {
        std::uint32_t qh;
        std::memcpy(&qh, w[b].qh, 4);
        std::int32_t acc = 0;
        for (std::size_t i = 0; i < kBlockSize / 2; ++i) {
            const int lo = ((w[b].qs[i] & 0x0F) | static_cast<int>(((qh >> i) & 1) << 4)) - 16;
            const int hi = ((w[b].qs[i] >> 4) | static_cast<int>(((qh >> (i + 16)) & 1) << 4)) - 16;
            acc += lo * x[b].qs[i] + hi * x[b].qs[i + kBlockSize / 2];
        }
        sum += static_cast<float>(acc) * half_to_float(w[b].d) * half_to_float(x[b].d);
    }
    return sum;
}

float dot_q8_0_q8_0(const BlockQ8_0* w, const BlockQ8_0* x, std::size_t blocks) {
    float sum = 0.0f;
    for (std::size_t b = 0; b < blocks; ++b) {
        std::int32_t acc = 0;
        for (std::size_t i = 0; i < kBlockSize; ++i) acc += w[b].qs[i] * x[b].qs[i];
        sum += static_cast<float>(acc) * half_to_float(w[b].d) * half_to_float(x[b].d);
    }
    return sum;
}

float dot_q4_1_q8_0(const BlockQ4_1* w, const BlockQ8_0* x, std::size_t blocks) {
    float sum = 0.0f;
    for (std::size_t b = 0; b < blocks; ++b) {
        std::int32_t acc = 0;
        std::int32_t xsum = 0;
        for (std::size_t i = 0; i < kBlockSize / 2; ++i) {
            acc += (w[b].qs[i] & 0x0F) * x[b].qs[i];
            acc += (w[b].qs[i] >> 4) * x[b].qs[i + kBlockSize / 2];
            xsum += x[b].qs[i] + x[b].qs[i + kBlockSize / 2];
        }
        const float dx = half_to_float(x[b].d);
        sum += static_cast<float>(acc) * half_to_float(w[b].d) * dx + half_to_float(w[b].m) * (dx * static_cast<float>(xsum));
    }
    return sum;
}

float dot_q4_k_q8_k(const BlockQ4_K* w, const BlockQ8_K* x, std::size_t blocks) {
    float sum = 0.0f;
    for (std::size_t b = 0; b < blocks; ++b) {
        const BlockQ4_K& wb = w[b];
        const BlockQ8_K& xb = x[b];
        std::int32_t weighted = 0;  // sum over sub-blocks of scale * dot
        std::int32_t offsets = 0;   // sum over sub-blocks of min * sum(x)
        for (int sb = 0; sb < 8; ++sb) {
            std::uint8_t scale, min;
            q4_k_scale_min(wb.scales, sb, scale, min);
            // Sub-block sb lives in chunk sb/2: low nibbles if even, high if odd.
            const std::uint8_t* q = wb.qs + (sb / 2) * 32;
            const std::int8_t* xq = xb.qs + sb * 32;
            std::int32_t dot = 0;
            for (int i = 0; i < 32; ++i) dot += ((sb % 2 == 0) ? (q[i] & 0xF) : (q[i] >> 4)) * xq[i];
            weighted += scale * dot;
            offsets += min * (xb.bsums[2 * sb] + xb.bsums[2 * sb + 1]);
        }
        sum += xb.d * (half_to_float(wb.d) * static_cast<float>(weighted) - half_to_float(wb.dmin) * static_cast<float>(offsets));
    }
    return sum;
}

float dot_q5_k_q8_k(const BlockQ5_K* w, const BlockQ8_K* x, std::size_t blocks) {
    float sum = 0.0f;
    for (std::size_t b = 0; b < blocks; ++b) {
        const BlockQ5_K& wb = w[b];
        const BlockQ8_K& xb = x[b];
        std::int32_t weighted = 0;
        std::int32_t offsets = 0;
        for (int sb = 0; sb < 8; ++sb) {
            std::uint8_t scale, min;
            q4_k_scale_min(wb.scales, sb, scale, min);
            const std::uint8_t* q = wb.qs + (sb / 2) * 32;
            const std::uint8_t bit = static_cast<std::uint8_t>(1u << sb);
            const std::int8_t* xq = xb.qs + sb * 32;
            std::int32_t dot = 0;
            for (int i = 0; i < 32; ++i) {
                const int v = ((sb % 2 == 0) ? (q[i] & 0xF) : (q[i] >> 4)) | ((wb.qh[i] & bit) ? 16 : 0);
                dot += v * xq[i];
            }
            weighted += scale * dot;
            offsets += min * (xb.bsums[2 * sb] + xb.bsums[2 * sb + 1]);
        }
        sum += xb.d * (half_to_float(wb.d) * static_cast<float>(weighted) - half_to_float(wb.dmin) * static_cast<float>(offsets));
    }
    return sum;
}

float dot_q6_k_q8_k(const BlockQ6_K* w, const BlockQ8_K* x, std::size_t blocks) {
    std::int8_t q[kSuperBlockSize];
    float sum = 0.0f;
    for (std::size_t b = 0; b < blocks; ++b) {
        unpack_q6_k(w[b], q);
        std::int32_t weighted = 0;
        for (int g = 0; g < 16; ++g) {
            std::int32_t dot = 0;
            for (int i = 0; i < 16; ++i) dot += q[g * 16 + i] * x[b].qs[g * 16 + i];
            weighted += w[b].scales[g] * dot;
        }
        sum += x[b].d * half_to_float(w[b].d) * static_cast<float>(weighted);
    }
    return sum;
}

void dequantize_tq2_0(const BlockTQ2_0* blocks, float* out, std::size_t count) {
    for (std::size_t b = 0; b < count / kSuperBlockSize; ++b) {
        const float d = half_to_float(blocks[b].d);
        float* y = out + b * kSuperBlockSize;
        for (std::size_t j = 0; j < 64; j += 32) {
            for (std::size_t l = 0; l < 4; ++l) {
                for (std::size_t m = 0; m < 32; ++m) {
                    const int q = (blocks[b].qs[j + m] >> (2 * l)) & 3;
                    *y++ = static_cast<float>(q - 1) * d;
                }
            }
        }
    }
}

float dot_tq2_0_q8_k(const BlockTQ2_0* w, const BlockQ8_K* x, std::size_t blocks) {
    float sum = 0.0f;
    for (std::size_t b = 0; b < blocks; ++b) {
        // sum((q - 1) x) = sum(q x) - sum(x), with sum(x) from the block sums.
        std::int32_t acc = 0;
        std::size_t i = 0;
        for (std::size_t j = 0; j < 64; j += 32) {
            for (std::size_t l = 0; l < 4; ++l) {
                for (std::size_t m = 0; m < 32; ++m) acc += ((w[b].qs[j + m] >> (2 * l)) & 3) * x[b].qs[i++];
            }
        }
        std::int32_t xsum = 0;
        for (const std::int16_t s : x[b].bsums) xsum += s;
        sum += x[b].d * half_to_float(w[b].d) * static_cast<float>(acc - xsum);
    }
    return sum;
}

namespace {

// Value 0..2 of element e of a TQ2_0 block.
inline int tq2_0_value(const BlockTQ2_0& block, std::size_t e) {
    const std::size_t j = (e / 128) * 32, l = (e / 32) % 4, m = e % 32;
    return (block.qs[j + m] >> (2 * l)) & 3;
}

// Byte and bit shift of element e (0..255) of row r (0..3) in a TQ2_0x4 block.
inline void tq2_0x4_position(std::size_t r, std::size_t e, std::size_t& byte, unsigned& shift) {
    const std::size_t chunk = e / 8;  // 0..31
    const std::size_t v = chunk / 4, l = chunk % 4;
    byte = 128 * (r / 2) + 16 * v + 8 * (r % 2) + (e % 8);
    shift = static_cast<unsigned>(2 * l);
}

}  // namespace

void repack_tq2_0x4(const BlockTQ2_0* in, std::size_t rows, std::size_t blocks, BlockTQ2_0x4* out) {
    for (std::size_t g = 0; g < rows / 4; ++g) {
        for (std::size_t b = 0; b < blocks; ++b) {
            BlockTQ2_0x4& packed = out[g * blocks + b];
            std::memset(packed.qs, 0, sizeof(packed.qs));
            for (std::size_t r = 0; r < 4; ++r) {
                const BlockTQ2_0& block = in[(g * 4 + r) * blocks + b];
                packed.d[r] = block.d;
                for (std::size_t e = 0; e < kSuperBlockSize; ++e) {
                    std::size_t byte;
                    unsigned shift;
                    tq2_0x4_position(r, e, byte, shift);
                    packed.qs[byte] |= static_cast<std::uint8_t>(tq2_0_value(block, e) << shift);
                }
            }
        }
    }
}

// The 128 pair indices (w0 * 3 + w1) of one TQ2_0 block, in column order.
void tq2_0_pair_indices(const BlockTQ2_0& block, std::uint8_t idx[128]) {
    // Bytes j + m (j in {0, 32}) hold elements j * 4 + 32 l + m at bits 2l: 32 consecutive
    // elements per (j, l), so pairs are adjacent lanes of one decoded vector.
    for (int j = 0; j < 2; ++j) {
        for (int l = 0; l < 4; ++l) {
            std::uint8_t codes[32];
#if defined(__aarch64__)
            const uint8x16_t mask = vdupq_n_u8(3);
            for (int h = 0; h < 2; ++h) {
                const uint8x16_t v = vld1q_u8(block.qs + 32 * j + 16 * h);
                const uint8x16_t c = vandq_u8(l == 0 ? v : l == 1 ? vshrq_n_u8(v, 2) : l == 2 ? vshrq_n_u8(v, 4) : vshrq_n_u8(v, 6), mask);
                vst1q_u8(codes + 16 * h, c);
            }
#else
            for (int m = 0; m < 32; ++m) codes[m] = static_cast<std::uint8_t>((block.qs[32 * j + m] >> (2 * l)) & 3);
#endif
            std::uint8_t* out = idx + (j * 128 + 32 * l) / 2;  // these 32 elements start at j * 128 + 32 l
            for (int m = 0; m < 32; m += 2) out[m / 2] = static_cast<std::uint8_t>(codes[m] * 3 + codes[m + 1]);
        }
    }
}

void repack_tq2_0x16(const BlockTQ2_0* in, std::size_t rows, std::size_t blocks, BlockTQ2_0x16* out) {
    for (std::size_t g = 0; g < rows / 16; ++g) {
        for (std::size_t b = 0; b < blocks; ++b) {
            BlockTQ2_0x16& packed = out[g * blocks + b];
            std::memset(packed.qs, 0, sizeof(packed.qs));
            for (std::size_t r = 0; r < 16; ++r) {
                const BlockTQ2_0& block = in[(g * 16 + r) * blocks + b];
                packed.d[r] = block.d;
                std::uint8_t idx[128];
                tq2_0_pair_indices(block, idx);
                for (std::size_t p = 0; p < 128; p += 2) packed.qs[(p / 2) * 16 + r] = static_cast<std::uint8_t>(idx[p] | (idx[p + 1] << 4));
            }
        }
    }
}

namespace {
inline int tq2_0x16_pair_index(const BlockTQ2_0x16& block, std::size_t r, std::size_t p) {
    return (block.qs[(p / 2) * 16 + r] >> (4 * (p % 2))) & 0xF;
}
}  // namespace

void dequantize_tq2_0x16_row(const BlockTQ2_0x16* group, std::size_t row_in_group, float* out, std::size_t count) {
    for (std::size_t b = 0; b < count / kSuperBlockSize; ++b) {
        const float d = half_to_float(group[b].d[row_in_group]);
        for (std::size_t p = 0; p < 128; ++p) {
            const int idx = tq2_0x16_pair_index(group[b], row_in_group, p);
            out[b * kSuperBlockSize + 2 * p] = static_cast<float>(idx / 3 - 1) * d;
            out[b * kSuperBlockSize + 2 * p + 1] = static_cast<float>(idx % 3 - 1) * d;
        }
    }
}

void dot_tq2_0x16_q8_k(const BlockTQ2_0x16* w, const BlockQ8_K* x, std::size_t blocks, float out[16]) {
    for (std::size_t r = 0; r < 16; ++r) {
        float sum = 0.0f;
        for (std::size_t b = 0; b < blocks; ++b) {
            std::int32_t acc = 0;
            for (std::size_t p = 0; p < 128; ++p) {
                const int idx = tq2_0x16_pair_index(w[b], r, p);
                acc += (idx / 3 - 1) * x[b].qs[2 * p] + (idx % 3 - 1) * x[b].qs[2 * p + 1];
            }
            sum += static_cast<float>(acc) * half_to_float(w[b].d[r]) * x[b].d;
        }
        out[r] = sum;
    }
}

void dequantize_tq2_0x4_row(const BlockTQ2_0x4* group, std::size_t row_in_group, float* out, std::size_t count) {
    for (std::size_t b = 0; b < count / kSuperBlockSize; ++b) {
        const float d = half_to_float(group[b].d[row_in_group]);
        for (std::size_t e = 0; e < kSuperBlockSize; ++e) {
            std::size_t byte;
            unsigned shift;
            tq2_0x4_position(row_in_group, e, byte, shift);
            out[b * kSuperBlockSize + e] = static_cast<float>(((group[b].qs[byte] >> shift) & 3) - 1) * d;
        }
    }
}

void dot_tq2_0x4_q8_k(const BlockTQ2_0x4* w, const BlockQ8_K* x, std::size_t blocks, float out[4]) {
    for (std::size_t r = 0; r < 4; ++r) {
        float sum = 0.0f;
        for (std::size_t b = 0; b < blocks; ++b) {
            std::int32_t acc = 0;
            for (std::size_t e = 0; e < kSuperBlockSize; ++e) {
                std::size_t byte;
                unsigned shift;
                tq2_0x4_position(r, e, byte, shift);
                acc += (((w[b].qs[byte] >> shift) & 3) - 1) * x[b].qs[e];
            }
            sum += static_cast<float>(acc) * half_to_float(w[b].d[r]) * x[b].d;
        }
        out[r] = sum;
    }
}

namespace {

// Value 0..15 of element e of a Q4_K block, and its sub-block.
inline int q4_k_value(const BlockQ4_K& block, std::size_t e) {
    const std::size_t chunk64 = e / 64, within = e % 64;
    const std::uint8_t byte = block.qs[chunk64 * 32 + within % 32];
    return within < 32 ? (byte & 0x0F) : (byte >> 4);
}

// Where element e of row r lives in a Q4_Kx4 block.
inline void q4_kx4_position(std::size_t r, std::size_t e, std::size_t& byte, bool& high) {
    const std::size_t sb = e / 32, c = (e % 32) / 8, k = e % 8;
    byte = 256 * (r / 2) + 32 * sb + 16 * (c % 2) + 8 * (r % 2) + k;
    high = c >= 2;
}

// Where element e of row r lives in a Q6_Kx4 block: the nibble and the bit pair.
inline void q6_kx4_position(std::size_t r, std::size_t e, std::size_t& low_byte, bool& low_high, std::size_t& high_byte,
                            unsigned& high_shift) {
    const std::size_t g = e / 64, c = (e % 64) / 8, k = e % 8;
    const std::size_t rp = r / 2, half = 8 * (r % 2);
    low_byte = (rp * 4 + g) * 64 + 16 * (c % 4) + half + k;
    low_high = c >= 4;
    high_byte = (rp * 4 + g) * 32 + 16 * (c / 4) + half + k;
    high_shift = static_cast<unsigned>(2 * (c % 4));
}

}  // namespace

void repack_q4_kx4(const BlockQ4_K* in, std::size_t rows, std::size_t blocks, BlockQ4_Kx4* out) {
    for (std::size_t g = 0; g < rows / 4; ++g) {
        for (std::size_t b = 0; b < blocks; ++b) {
            BlockQ4_Kx4& packed = out[g * blocks + b];
            std::memset(packed.qs, 0, sizeof(packed.qs));
            for (std::size_t r = 0; r < 4; ++r) {
                const BlockQ4_K& block = in[(g * 4 + r) * blocks + b];
                packed.d[r] = block.d;
                packed.dmin[r] = block.dmin;
                std::memcpy(packed.scales[r], block.scales, 12);
                for (std::size_t e = 0; e < kSuperBlockSize; ++e) {
                    std::size_t byte;
                    bool high;
                    q4_kx4_position(r, e, byte, high);
                    const int q = q4_k_value(block, e);
                    packed.qs[byte] |= static_cast<std::uint8_t>(high ? q << 4 : q);
                }
            }
        }
    }
}

void dequantize_q4_kx4_row(const BlockQ4_Kx4* group, std::size_t row_in_group, float* out, std::size_t count) {
    for (std::size_t b = 0; b < count / kSuperBlockSize; ++b) {
        const BlockQ4_Kx4& packed = group[b];
        const float d = half_to_float(packed.d[row_in_group]);
        const float dmin = half_to_float(packed.dmin[row_in_group]);
        for (std::size_t e = 0; e < kSuperBlockSize; ++e) {
            std::uint8_t scale, min;
            q4_k_scale_min(packed.scales[row_in_group], static_cast<int>(e / 32), scale, min);
            std::size_t byte;
            bool high;
            q4_kx4_position(row_in_group, e, byte, high);
            const int q = high ? (packed.qs[byte] >> 4) : (packed.qs[byte] & 0x0F);
            out[b * kSuperBlockSize + e] = d * scale * static_cast<float>(q) - dmin * min;
        }
    }
}

void dot_q4_kx4_q8_k(const BlockQ4_Kx4* w, const BlockQ8_K* x, std::size_t blocks, float out[4]) {
    for (std::size_t r = 0; r < 4; ++r) {
        float sum = 0.0f;
        for (std::size_t b = 0; b < blocks; ++b) {
            std::int32_t weighted = 0, offsets = 0;
            for (int sb = 0; sb < 8; ++sb) {
                std::uint8_t scale, min;
                q4_k_scale_min(w[b].scales[r], sb, scale, min);
                std::int32_t dot = 0;
                for (std::size_t i = 0; i < 32; ++i) {
                    const std::size_t e = static_cast<std::size_t>(sb) * 32 + i;
                    std::size_t byte;
                    bool high;
                    q4_kx4_position(r, e, byte, high);
                    dot += (high ? (w[b].qs[byte] >> 4) : (w[b].qs[byte] & 0x0F)) * x[b].qs[e];
                }
                weighted += scale * dot;
                offsets += min * (x[b].bsums[2 * sb] + x[b].bsums[2 * sb + 1]);
            }
            sum += x[b].d * (half_to_float(w[b].d[r]) * static_cast<float>(weighted) -
                             half_to_float(w[b].dmin[r]) * static_cast<float>(offsets));
        }
        out[r] = sum;
    }
}

void repack_q6_kx4(const BlockQ6_K* in, std::size_t rows, std::size_t blocks, BlockQ6_Kx4* out) {
    std::int8_t q[kSuperBlockSize];
    for (std::size_t g = 0; g < rows / 4; ++g) {
        for (std::size_t b = 0; b < blocks; ++b) {
            BlockQ6_Kx4& packed = out[g * blocks + b];
            std::memset(packed.low, 0, sizeof(packed.low));
            std::memset(packed.high, 0, sizeof(packed.high));
            for (std::size_t r = 0; r < 4; ++r) {
                const BlockQ6_K& block = in[(g * 4 + r) * blocks + b];
                packed.d[r] = block.d;
                std::memcpy(packed.scales[r], block.scales, 16);
                unpack_q6_k(block, q);  // values -32..31
                for (std::size_t e = 0; e < kSuperBlockSize; ++e) {
                    const unsigned v = static_cast<unsigned>(q[e] + 32);  // 0..63
                    std::size_t low_byte, high_byte;
                    bool low_high;
                    unsigned high_shift;
                    q6_kx4_position(r, e, low_byte, low_high, high_byte, high_shift);
                    (&packed.low[0][0][0])[low_byte] |= static_cast<std::uint8_t>(low_high ? (v & 0xF) << 4 : (v & 0xF));
                    (&packed.high[0][0][0])[high_byte] |= static_cast<std::uint8_t>((v >> 4) << high_shift);
                }
            }
        }
    }
}

namespace {

inline int q6_kx4_value(const BlockQ6_Kx4& packed, std::size_t r, std::size_t e) {
    std::size_t low_byte, high_byte;
    bool low_high;
    unsigned high_shift;
    q6_kx4_position(r, e, low_byte, low_high, high_byte, high_shift);
    const std::uint8_t lo = (&packed.low[0][0][0])[low_byte];
    const std::uint8_t hi = (&packed.high[0][0][0])[high_byte];
    return static_cast<int>((low_high ? (lo >> 4) : (lo & 0xF)) | (((hi >> high_shift) & 3) << 4)) - 32;
}

}  // namespace

void dequantize_q6_kx4_row(const BlockQ6_Kx4* group, std::size_t row_in_group, float* out, std::size_t count) {
    for (std::size_t b = 0; b < count / kSuperBlockSize; ++b) {
        const float d = half_to_float(group[b].d[row_in_group]);
        for (std::size_t e = 0; e < kSuperBlockSize; ++e) {
            out[b * kSuperBlockSize + e] =
                d * group[b].scales[row_in_group][e / 16] * static_cast<float>(q6_kx4_value(group[b], row_in_group, e));
        }
    }
}

void dot_q6_kx4_q8_k(const BlockQ6_Kx4* w, const BlockQ8_K* x, std::size_t blocks, float out[4]) {
    for (std::size_t r = 0; r < 4; ++r) {
        float sum = 0.0f;
        for (std::size_t b = 0; b < blocks; ++b) {
            std::int32_t weighted = 0;
            for (int sb = 0; sb < 16; ++sb) {
                std::int32_t dot = 0;
                for (std::size_t i = 0; i < 16; ++i) {
                    const std::size_t e = static_cast<std::size_t>(sb) * 16 + i;
                    dot += q6_kx4_value(w[b], r, e) * x[b].qs[e];
                }
                weighted += w[b].scales[r][sb] * dot;
            }
            sum += x[b].d * half_to_float(w[b].d[r]) * static_cast<float>(weighted);
        }
        out[r] = sum;
    }
}

}  // namespace brisk
