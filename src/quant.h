#pragma once

#include <cstddef>
#include <cstdint>

// Quantised block formats, laid out exactly as in GGUF/ggml so weights can be
// used straight from the file.

namespace brisk {

constexpr std::size_t kBlockSize = 32;  // weights per block for Q4_0 and Q8_0

// Storage formats of weight matrices that the engine can multiply by.
enum class WeightFormat { F32, Q4_0, Q4_1, Q4_K, Q6_K };

// Weights per block and bytes per block of a quantised format.
std::size_t block_elements(WeightFormat format);
std::size_t block_bytes(WeightFormat format);
// True when the format's kernels take Q8_K activations rather than Q8_0.
bool takes_q8_k(WeightFormat format);

// 32 weights as 4-bit values in [0, 16) with a shared half-precision scale:
// weight = (q - 8) * d. The low nibbles hold weights 0-15, the high nibbles 16-31.
struct BlockQ4_0 {
    std::uint16_t d;
    std::uint8_t qs[kBlockSize / 2];
};
static_assert(sizeof(BlockQ4_0) == 18);

// 32 values as signed 8-bit with a shared half-precision scale: value = q * d.
// Activations are quantised to this format before the integer dot products.
struct BlockQ8_0 {
    std::uint16_t d;
    std::int8_t qs[kBlockSize];
};
static_assert(sizeof(BlockQ8_0) == 34);

// 32 weights as 4-bit values with a scale and an offset: weight = q * d + m.
struct BlockQ4_1 {
    std::uint16_t d;
    std::uint16_t m;
    std::uint8_t qs[kBlockSize / 2];
};
static_assert(sizeof(BlockQ4_1) == 20);

constexpr std::size_t kSuperBlockSize = 256;  // weights per block for the K-quants

// 256 weights in 8 sub-blocks of 32. Each sub-block has a 6-bit scale and
// 6-bit minimum packed into `scales`; weight = d * scale * q - dmin * min.
struct BlockQ4_K {
    std::uint16_t d;
    std::uint16_t dmin;
    std::uint8_t scales[12];
    std::uint8_t qs[kSuperBlockSize / 2];
};
static_assert(sizeof(BlockQ4_K) == 144);

// 256 weights as 6-bit values (low 4 bits in ql, high 2 in qh) in 16
// sub-blocks of 16 with signed 8-bit scales: weight = d * scale * (q - 32).
struct BlockQ6_K {
    std::uint8_t ql[kSuperBlockSize / 2];
    std::uint8_t qh[kSuperBlockSize / 4];
    std::int8_t scales[kSuperBlockSize / 16];
    std::uint16_t d;
};
static_assert(sizeof(BlockQ6_K) == 210);

// Activations for the K-quants: 256 signed 8-bit values with a float scale,
// plus the sum of each group of 16, which the kernels need for the offsets.
struct BlockQ8_K {
    float d;
    std::int8_t qs[kSuperBlockSize];
    std::int16_t bsums[kSuperBlockSize / 16];
};
static_assert(sizeof(BlockQ8_K) == 292);

float half_to_float(std::uint16_t h);
std::uint16_t float_to_half(float f);  // round to nearest, ties to even

// `count` must be a multiple of the format's block size.
void quantize_q8_0(const float* x, BlockQ8_0* out, std::size_t count);
void quantize_q8_k(const float* x, BlockQ8_K* out, std::size_t count);
void dequantize_q4_0(const BlockQ4_0* blocks, float* out, std::size_t count);
void dequantize_q4_1(const BlockQ4_1* blocks, float* out, std::size_t count);
void dequantize_q4_k(const BlockQ4_K* blocks, float* out, std::size_t count);
void dequantize_q6_k(const BlockQ6_K* blocks, float* out, std::size_t count);

template <typename Block>
constexpr std::size_t block_elements_of() {
    return sizeof(Block) == sizeof(BlockQ4_0) || sizeof(Block) == sizeof(BlockQ4_1) ? kBlockSize : kSuperBlockSize;
}

// Decodes the 256 6-bit values of a Q6_K block, offset removed, in weight order.
void unpack_q6_k(const BlockQ6_K& block, std::int8_t* q);

// Dot products of one weight row with quantised activations, over `blocks` blocks.
float dot_q4_0_q8_0(const BlockQ4_0* w, const BlockQ8_0* x, std::size_t blocks);
float dot_q4_1_q8_0(const BlockQ4_1* w, const BlockQ8_0* x, std::size_t blocks);
float dot_q4_k_q8_k(const BlockQ4_K* w, const BlockQ8_K* x, std::size_t blocks);
float dot_q6_k_q8_k(const BlockQ6_K* w, const BlockQ8_K* x, std::size_t blocks);

// Unpacks the 6-bit scale and minimum of sub-block j (0-7) of a Q4_K block.
inline void q4_k_scale_min(const std::uint8_t* scales, int j, std::uint8_t& scale, std::uint8_t& min) {
    if (j < 4) {
        scale = scales[j] & 63;
        min = scales[j + 4] & 63;
    } else {
        scale = static_cast<std::uint8_t>((scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4));
        min = static_cast<std::uint8_t>((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4));
    }
}

}  // namespace brisk
