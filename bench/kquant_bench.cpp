// Batched (64-token) throughput of the K-quant CPU kernels against Q4_0's, one
// thread, in multiply-adds per second: the prompt-processing rate per core.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "kernels.h"
#include "quant.h"

namespace {
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }
}  // namespace

int main(int argc, char** argv) {
    const std::size_t rows = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1024;
    const std::size_t cols = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 2048;
    constexpr std::size_t kTokens = 64;
    std::mt19937 rng(1);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> x(kTokens * cols);
    for (float& v : x) v = normal(rng);

    for (const brisk::WeightFormat format : {brisk::WeightFormat::Q4_0, brisk::WeightFormat::Q4_K, brisk::WeightFormat::Q6_K}) {
        const std::size_t blocks = cols / brisk::block_elements(format);
        std::vector<std::uint8_t> w(rows * blocks * brisk::block_bytes(format));
        for (std::uint8_t& b : w) b = static_cast<std::uint8_t>(rng());
        for (std::size_t b = 0; b < rows * blocks; ++b) {  // sane scales
            std::uint8_t* blk = w.data() + b * brisk::block_bytes(format);
            const std::uint16_t d = brisk::float_to_half(0.01f);
            if (format == brisk::WeightFormat::Q6_K) std::memcpy(blk + 208, &d, 2);
            else { std::memcpy(blk, &d, 2); if (format == brisk::WeightFormat::Q4_K) std::memcpy(blk + 2, &d, 2); }
        }
        const bool k = brisk::takes_q8_k(format);
        std::vector<std::uint8_t> xq(kTokens * blocks * (k ? sizeof(brisk::BlockQ8_K) : sizeof(brisk::BlockQ8_0)));
        for (std::size_t t = 0; t < kTokens; ++t) {
            if (k) brisk::quantize_q8_k(x.data() + t * cols, reinterpret_cast<brisk::BlockQ8_K*>(xq.data()) + t * blocks, cols);
            else brisk::quantize_q8_0(x.data() + t * cols, reinterpret_cast<brisk::BlockQ8_0*>(xq.data()) + t * blocks, cols);
        }
        std::vector<float> out(kTokens * rows);
        for (const brisk::kernels::NamedKernels& nk : brisk::kernels::neon_kernels(format)) {
            std::vector<std::uint8_t> prepared;
            const void* input = xq.data();
            if (nk.kernels.prepare != nullptr) {
                prepared.resize(nk.kernels.prepared_bytes(kTokens, blocks));
                nk.kernels.prepare(xq.data(), kTokens, blocks, prepared.data());
                input = prepared.data();
            }
            int reps = 0;
            const auto start = Clock::now();
            while (seconds(start) < 1.0) {
                nk.kernels.matmul(w.data(), rows, cols, input, kTokens, out.data(), rows);
                ++reps;
            }
            const double macs = static_cast<double>(rows) * cols * kTokens * reps;
            std::printf("%-5s %-8.*s batched %5.2f GMAC/s\n", format == brisk::WeightFormat::Q4_0 ? "Q4_0" : format == brisk::WeightFormat::Q4_K ? "Q4_K" : "Q6_K",
                        static_cast<int>(nk.name.size()), nk.name.data(), macs / seconds(start) / 1e9);
        }
    }
}
