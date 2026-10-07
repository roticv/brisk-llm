#include "quant.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace brisk {

std::size_t block_elements(WeightFormat format) {
    switch (format) {
        case WeightFormat::F32: return 1;
        case WeightFormat::Q4_0:
        case WeightFormat::Q4_1: return kBlockSize;
        case WeightFormat::Q4_K:
        case WeightFormat::Q6_K: return kSuperBlockSize;
    }
    return 1;
}

std::size_t block_bytes(WeightFormat format) {
    switch (format) {
        case WeightFormat::F32: return sizeof(float);
        case WeightFormat::Q4_0: return sizeof(BlockQ4_0);
        case WeightFormat::Q4_1: return sizeof(BlockQ4_1);
        case WeightFormat::Q4_K: return sizeof(BlockQ4_K);
        case WeightFormat::Q6_K: return sizeof(BlockQ6_K);
    }
    return 0;
}

bool takes_q8_k(WeightFormat format) { return format == WeightFormat::Q4_K || format == WeightFormat::Q6_K; }

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
        for (std::size_t i = 0; i < kBlockSize; ++i) {
            out[b].qs[i] = static_cast<std::int8_t>(std::nearbyint(block[i] * inverse));
        }
    }
}

void quantize_q8_k(const float* x, BlockQ8_K* out, std::size_t count) {
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
        const float inverse = -127.0f / max;
        y.d = 1.0f / inverse;
        for (std::size_t i = 0; i < kSuperBlockSize; ++i) {
            y.qs[i] = static_cast<std::int8_t>(std::min(127.0f, std::nearbyint(inverse * block[i])));
        }
        for (std::size_t g = 0; g < kSuperBlockSize / 16; ++g) {
            int sum = 0;
            for (std::size_t i = 0; i < 16; ++i) sum += y.qs[g * 16 + i];
            y.bsums[g] = static_cast<std::int16_t>(sum);
        }
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

}  // namespace brisk
