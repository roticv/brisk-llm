// Measures the Q4_0 kernels on one thread, in isolation from the rest of the
// engine: single-row (memory-bound) and batched (compute-bound) throughput.
//
// Built as part of the normal CMake build. Usage: kernel_bench [rows] [cols]

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string_view>
#include <vector>

#include "kernels.h"
#include "quant.h"

namespace {

using Clock = std::chrono::steady_clock;

double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }

}  // namespace

int main(int argc, char** argv) {
    const std::size_t rows = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 2048;
    const std::size_t cols = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 1024;
    const std::size_t blocks = cols / brisk::kBlockSize;

    std::mt19937 rng(1);
    std::vector<brisk::BlockQ4_0> w(rows * blocks);
    for (brisk::BlockQ4_0& b : w) {
        b.d = brisk::float_to_half(0.01f);
        for (std::uint8_t& q : b.qs) q = static_cast<std::uint8_t>(rng());
    }
    constexpr std::size_t kTokens = 64;
    std::vector<float> x(kTokens * cols);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (float& v : x) v = normal(rng);
    std::vector<brisk::BlockQ8_0> xq(kTokens * blocks);
    for (std::size_t t = 0; t < kTokens; ++t) brisk::quantize_q8_0(x.data() + t * cols, xq.data() + t * blocks, cols);
    std::vector<float> out(kTokens * rows);

    const double weight_mb = static_cast<double>(w.size() * sizeof(brisk::BlockQ4_0)) / 1e6;
    std::vector<brisk::BlockQ4_0x4> packed(rows / 4 * blocks);
    brisk::repack_q4_0x4(w.data(), rows / 4 * 4, blocks, packed.data());
    std::printf("%zu x %zu Q4_0 matrix (%.1f MB), one thread\n", rows, cols, weight_mb);

    struct Variant {
        std::string_view name;
        brisk::WeightFormat format;
        brisk::kernels::KernelSet kernels;
    };
    std::vector<Variant> variants{{"generic", brisk::WeightFormat::Q4_0, brisk::kernels::generic_kernels(brisk::WeightFormat::Q4_0)}};
#if defined(__aarch64__)
    for (const brisk::WeightFormat format : {brisk::WeightFormat::Q4_0, brisk::WeightFormat::Q4_0x4}) {
        for (const brisk::kernels::NamedKernels& k : brisk::kernels::neon_kernels(format)) {
            variants.push_back({k.name, format, k.kernels});
        }
    }
#endif
    for (const Variant& v : variants) {
        const void* weights = v.format == brisk::WeightFormat::Q4_0x4 ? static_cast<const void*>(packed.data())
                                                                     : static_cast<const void*>(w.data());
        // Single row: report weight bytes streamed per second.
        int reps = 0;
        auto start = Clock::now();
        while (seconds(start) < 0.5) {
            v.kernels.matvec(weights, rows, cols, xq.data(), out.data());
            ++reps;
        }
        const double single_gbps = weight_mb * reps / seconds(start) / 1e3;

        // Batched: report multiply-adds per second (activation preparation included).
        std::vector<std::uint8_t> prepared;
        reps = 0;
        start = Clock::now();
        while (seconds(start) < 1.0) {
            const void* input = xq.data();
            if (v.kernels.prepare != nullptr) {
                prepared.resize(v.kernels.prepared_bytes(kTokens, blocks));
                v.kernels.prepare(xq.data(), kTokens, blocks, prepared.data());
                input = prepared.data();
            }
            v.kernels.matmul(weights, rows, cols, input, kTokens, out.data());
            ++reps;
        }
        const double gmacs = static_cast<double>(rows * cols * kTokens) * reps / seconds(start) / 1e9;
        std::printf("%-8.*s %-7s single-row %6.2f GB/s   batched (%zu tokens) %6.2f GMAC/s\n", static_cast<int>(v.name.size()),
                    v.name.data(), v.format == brisk::WeightFormat::Q4_0x4 ? "Q4_0x4" : "Q4_0", single_gbps, kTokens, gmacs);
    }
    return 0;
}
