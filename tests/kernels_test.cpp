#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string_view>
#include <vector>

#include "kernels.h"
#include "quant.h"

namespace {

using brisk::BlockQ4_0;
using brisk::BlockQ8_0;

#define CHECK(cond)                                                                       \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            std::exit(1);                                                                 \
        }                                                                                 \
    } while (0)

void test_half_conversion_round_trips() {
    for (std::uint32_t h = 0; h < 0x10000; ++h) {
        const auto half = static_cast<std::uint16_t>(h);
        const float f = brisk::half_to_float(half);
        if (std::isnan(f)) continue;
        CHECK(brisk::float_to_half(f) == half);
    }
    // Rounding: 1 + 2^-11 lies exactly between two halves and rounds to the even one.
    CHECK(brisk::float_to_half(1.0f + 1.0f / 2048) == 0x3C00);
    CHECK(brisk::float_to_half(1.0f + 3.0f / 2048) == 0x3C02);
    CHECK(brisk::float_to_half(65520.0f) == 0x7C00);  // rounds up to infinity
    CHECK(brisk::float_to_half(1e-8f) == 0);
}

void test_quantize_q8_0() {
    float x[32];
    for (int i = 0; i < 32; ++i) x[i] = static_cast<float>(i - 16) * 0.5f;  // amax 8 at i = 0
    BlockQ8_0 q;
    brisk::quantize_q8_0(x, &q, 32);
    CHECK(q.d == brisk::float_to_half(8.0f / 127));
    CHECK(q.qs[0] == -127 && q.qs[16] == 0 && q.qs[31] == 119);  // 7.5 / (8/127) = 119.06
}

// Random weights in every format, random activations, and a float reference
// computed from the dequantised weights: every kernel (generic and NEON,
// single-row and batched) must land within 8-bit activation rounding of it.
void test_kernels_match_dequantised_reference() {
    constexpr std::size_t rows = 37, cols = 1024, n = 7;
    std::mt19937 rng(7);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    auto byte = [&] { return static_cast<std::uint8_t>(rng()); };

    std::vector<float> x(n * cols);
    for (float& v : x) v = normal(rng);

    const brisk::WeightFormat formats[] = {brisk::WeightFormat::Q4_0, brisk::WeightFormat::Q4_1,
                                           brisk::WeightFormat::Q4_K, brisk::WeightFormat::Q6_K};
    for (const brisk::WeightFormat format : formats) {
        const std::size_t blocks = cols / brisk::block_elements(format);
        std::vector<std::uint8_t> w(rows * blocks * brisk::block_bytes(format));
        for (std::uint8_t& b : w) b = byte();
        // Keep the scales small and sane: overwrite each block's half-precision fields.
        for (std::size_t b = 0; b < rows * blocks; ++b) {
            std::uint8_t* block = w.data() + b * brisk::block_bytes(format);
            const std::uint16_t d = brisk::float_to_half(0.001f + 0.001f * std::fabs(normal(rng)));
            switch (format) {
                case brisk::WeightFormat::Q4_0: std::memcpy(block, &d, 2); break;
                case brisk::WeightFormat::Q4_1: std::memcpy(block, &d, 2); std::memcpy(block + 2, &d, 2); break;
                case brisk::WeightFormat::Q4_K: std::memcpy(block, &d, 2); std::memcpy(block + 2, &d, 2); break;
                case brisk::WeightFormat::Q6_K: std::memcpy(block + 208, &d, 2); break;
                case brisk::WeightFormat::F32: break;
            }
        }

        // Reference: dequantise and dot in float.
        std::vector<float> dequant(cols);
        std::vector<float> reference(n * rows);
        std::vector<float> row_norm(rows);
        for (std::size_t r = 0; r < rows; ++r) {
            const std::uint8_t* row = w.data() + r * blocks * brisk::block_bytes(format);
            switch (format) {
                case brisk::WeightFormat::Q4_0: brisk::dequantize_q4_0(reinterpret_cast<const BlockQ4_0*>(row), dequant.data(), cols); break;
                case brisk::WeightFormat::Q4_1: brisk::dequantize_q4_1(reinterpret_cast<const brisk::BlockQ4_1*>(row), dequant.data(), cols); break;
                case brisk::WeightFormat::Q4_K: brisk::dequantize_q4_k(reinterpret_cast<const brisk::BlockQ4_K*>(row), dequant.data(), cols); break;
                case brisk::WeightFormat::Q6_K: brisk::dequantize_q6_k(reinterpret_cast<const brisk::BlockQ6_K*>(row), dequant.data(), cols); break;
                case brisk::WeightFormat::F32: break;
            }
            double norm2 = 0.0;
            for (const float v : dequant) norm2 += static_cast<double>(v) * v;
            row_norm[r] = static_cast<float>(std::sqrt(norm2));
            for (std::size_t t = 0; t < n; ++t) {
                double sum = 0.0;
                for (std::size_t i = 0; i < cols; ++i) sum += static_cast<double>(dequant[i]) * x[t * cols + i];
                reference[t * rows + r] = static_cast<float>(sum);
            }
        }
        std::vector<float> x_max(n, 0.0f);
        for (std::size_t t = 0; t < n; ++t) {
            for (std::size_t i = 0; i < cols; ++i) x_max[t] = std::max(x_max[t], std::fabs(x[t * cols + i]));
        }

        // Activations in the format the kernels take.
        std::vector<BlockQ8_0> q8_0(n * blocks);
        std::vector<brisk::BlockQ8_K> q8_k(n * blocks);
        const void* activations = nullptr;
        std::size_t activation_bytes = 0;
        if (brisk::takes_q8_k(format)) {
            for (std::size_t t = 0; t < n; ++t) brisk::quantize_q8_k(x.data() + t * cols, q8_k.data() + t * blocks, cols);
            activations = q8_k.data();
            activation_bytes = blocks * sizeof(brisk::BlockQ8_K);
        } else {
            for (std::size_t t = 0; t < n; ++t) brisk::quantize_q8_0(x.data() + t * cols, q8_0.data() + t * blocks, cols);
            activations = q8_0.data();
            activation_bytes = blocks * sizeof(BlockQ8_0);
        }

        struct Variant {
            std::string_view name;
            brisk::kernels::KernelSet kernels;
        };
        std::vector<Variant> variants{{"generic", brisk::kernels::generic_kernels(format)}};
#if defined(__aarch64__)
        for (const brisk::kernels::NamedKernels& k : brisk::kernels::neon_kernels(format)) {
            variants.push_back({k.name, k.kernels});
        }
#endif
        for (const Variant& v : variants) {
            std::vector<float> batched(n * rows, 12345.0f);
            v.kernels.matmul(w.data(), rows, cols, activations, n, batched.data());
            for (std::size_t t = 0; t < n; ++t) {
                std::vector<float> single(rows, 12345.0f);
                const auto* xt = static_cast<const std::uint8_t*>(activations) + t * activation_bytes;
                v.kernels.matvec(w.data(), rows, cols, xt, single.data());
                for (std::size_t r = 0; r < rows; ++r) {
                    const float ref = reference[t * rows + r];
                    // Each 8-bit activation is off by up to half a step (max|x| / 254); over a
                    // row the errors add like a random walk, so the bound is the row's 2-norm
                    // times that step, with a 4x margin.
                    const float tolerance = 0.02f + 4.0f * row_norm[r] * x_max[t] / 254.0f;
                    if (std::fabs(single[r] - ref) > tolerance || std::fabs(batched[t * rows + r] - single[r]) > 1e-4f * (1.0f + std::fabs(ref))) {
                        std::fprintf(stderr, "format %d %.*s token %zu row %zu: single %f batched %f reference %f\n",
                                     static_cast<int>(format), static_cast<int>(v.name.size()), v.name.data(), t, r,
                                     single[r], batched[t * rows + r], ref);
                        std::exit(1);
                    }
                }
            }
            std::printf("kernels_test: format %d %.*s ok\n", static_cast<int>(format), static_cast<int>(v.name.size()),
                        v.name.data());
        }
    }
}

}  // namespace

int main() {
    test_half_conversion_round_trips();
    test_quantize_q8_0();
    test_kernels_match_dequantised_reference();
    std::printf("kernels_test: ok (selected: %.*s)\n", static_cast<int>(brisk::kernels::kernel_set_name().size()),
                brisk::kernels::kernel_set_name().data());
    return 0;
}
