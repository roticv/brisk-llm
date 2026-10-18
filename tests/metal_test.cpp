// Checks the Metal matmul kernels (gemv and 64x64 tile, each weight format)
// against a float reference on random blocks.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "metal/metal_session.h"
#include "quant.h"

using namespace brisk;

namespace {

const char* name(WeightFormat f) {
    switch (f) {
        case WeightFormat::Q4_0: return "q4_0";
        case WeightFormat::Q4_1: return "q4_1";
        case WeightFormat::Q5_0: return "q5_0";
        case WeightFormat::Q8_0: return "q8_0";
        case WeightFormat::Q4_K: return "q4_k";
        case WeightFormat::Q6_K: return "q6_k";
        default: return "?";
    }
}

// Random blocks of `format` for rows x cols, and their dequantised values.
std::vector<std::byte> random_blocks(WeightFormat format, std::size_t rows, std::size_t cols, std::mt19937& rng,
                                     std::vector<float>& values) {
    const std::size_t count = rows * cols;
    const std::size_t blocks = count / block_elements(format);
    std::vector<std::byte> bytes(blocks * block_bytes(format));
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> scale(0.02f, 0.1f);
    for (std::size_t b = 0; b < blocks; ++b) {
        std::byte* blk = bytes.data() + b * block_bytes(format);
        for (std::size_t i = 0; i < block_bytes(format); ++i) blk[i] = static_cast<std::byte>(byte(rng));
        // Sane scales: random bytes make inf/NaN halves.
        const std::uint16_t d = float_to_half(scale(rng));
        switch (format) {
            case WeightFormat::Q4_0: std::memcpy(blk, &d, 2); break;
            case WeightFormat::Q4_1: {
                const std::uint16_t m = float_to_half(scale(rng) - 0.06f);
                std::memcpy(blk, &d, 2);
                std::memcpy(blk + 2, &m, 2);
                break;
            }
            case WeightFormat::Q4_K: {
                const std::uint16_t dmin = float_to_half(scale(rng) * 0.1f);
                std::memcpy(blk, &d, 2);
                std::memcpy(blk + 2, &dmin, 2);
                break;
            }
            case WeightFormat::Q6_K: std::memcpy(blk + 208, &d, 2); break;
            case WeightFormat::Q5_0:
            case WeightFormat::Q8_0: std::memcpy(blk, &d, 2); break;
            default: break;
        }
    }
    values.resize(count);
    switch (format) {
        case WeightFormat::Q4_0: dequantize_q4_0(reinterpret_cast<const BlockQ4_0*>(bytes.data()), values.data(), count); break;
        case WeightFormat::Q4_1: dequantize_q4_1(reinterpret_cast<const BlockQ4_1*>(bytes.data()), values.data(), count); break;
        case WeightFormat::Q4_K: dequantize_q4_k(reinterpret_cast<const BlockQ4_K*>(bytes.data()), values.data(), count); break;
        case WeightFormat::Q6_K: dequantize_q6_k(reinterpret_cast<const BlockQ6_K*>(bytes.data()), values.data(), count); break;
        case WeightFormat::Q5_0: dequantize_q5_0(reinterpret_cast<const BlockQ5_0*>(bytes.data()), values.data(), count); break;
        case WeightFormat::Q8_0: dequantize_q8_0(reinterpret_cast<const BlockQ8_0*>(bytes.data()), values.data(), count); break;
        default: break;
    }
    return bytes;
}

bool check(WeightFormat format, std::size_t rows, std::size_t cols, std::size_t n, bool tiled, bool paired = false) {
    std::mt19937 rng(static_cast<unsigned>(rows * 31 + cols * 7 + n + static_cast<unsigned>(format) * 101));
    std::normal_distribution<float> dist;
    std::vector<float> x(n * cols);
    for (float& v : x) v = dist(rng);
    std::vector<float> w;
    const std::vector<std::byte> bytes = random_blocks(format, rows, cols, rng, w);

    // Reference: the dequantised weights against the exact float activations.
    std::vector<float> expected(rows * n);
    double scale = 0.0;  // typical magnitude of a weight, for the tolerance
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t t = 0; t < n; ++t) {
            double sum = 0.0;
            for (std::size_t c = 0; c < cols; ++c) sum += double(w[r * cols + c]) * x[t * cols + c];
            if (paired) sum = 0.5 * sum * (1.0 + std::tanh(0.7978845608 * (sum + 0.044715 * sum * sum * sum))) * sum;  // gelu(gate) * up, gate = up
            expected[t * rows + r] = static_cast<float>(sum);
        }
        for (std::size_t c = 0; c < cols; ++c) scale += std::fabs(w[r * cols + c]);
    }
    scale /= double(rows * cols);
    const std::vector<float> got = metal_matmul(format, bytes, rows, cols, x, n, tiled, paired);

    if (std::getenv("METAL_TEST_DUMP") != nullptr) {
        for (std::size_t i = 0; i < std::min<std::size_t>(8, expected.size()); ++i) {
            std::printf("  [%zu] expected %g got %g\n", i, expected[i], got[i]);
        }
    }
    double worst = 0.0;
    for (std::size_t i = 0; i < expected.size(); ++i) worst = std::max(worst, std::fabs(double(expected[i]) - got[i]));
    // Half activations in the tiled kernel: relative error ~1e-3 of a dot of `cols` terms of size `scale`.
    const double tolerance = (tiled ? 4e-3 : 1e-4) * std::sqrt(double(cols)) * std::max(scale, 0.3) * (paired ? 50.0 : 1.0);
    const bool ok = worst <= tolerance;
    std::printf("%s %s rows %zu cols %zu n %zu: max abs error %.3g (tolerance %.3g) %s\n", name(format),
                paired ? "gated" : tiled ? "tiled" : "gemv ", rows, cols, n, worst, tolerance, ok ? "ok" : "FAILED");
    return ok;
}

}  // namespace

int main() {
    if (!MetalSession::available()) {
        std::printf("metal_test: no Metal device, skipped\n");
        return 0;
    }
    bool ok = true;
    ok &= check(WeightFormat::Q4_0, 64, 64, 1, false);
    ok &= check(WeightFormat::Q4_0, 130, 2048, 3, false);
    ok &= check(WeightFormat::Q4_0, 1024, 2048, 7, false);
    ok &= check(WeightFormat::Q4_0, 130, 1152, 2, false);  // 36 blocks per row
    ok &= check(WeightFormat::Q4_0, 1152, 1024, 2, false);
    ok &= check(WeightFormat::Q4_0, 262144 / 64, 1152, 1, false);
    ok &= check(WeightFormat::Q4_0, 130, 1152, 2, false, true);  // the paired (gate/up) epilogue
    ok &= check(WeightFormat::Q4_K, 130, 1280, 2, false, true);
    ok &= check(WeightFormat::Q4_0, 128, 64, 8, true);
    ok &= check(WeightFormat::Q4_0, 128, 2048, 64, true);
    ok &= check(WeightFormat::Q4_0, 256, 1024, 100, true);
    ok &= check(WeightFormat::Q4_0, 384, 96, 130, true);
    for (const WeightFormat f : {WeightFormat::Q4_1, WeightFormat::Q5_0, WeightFormat::Q8_0, WeightFormat::Q4_K, WeightFormat::Q6_K}) {
        const std::size_t cols = (f == WeightFormat::Q4_1 || f == WeightFormat::Q5_0 || f == WeightFormat::Q8_0) ? 1056 : 1280;
        ok &= check(f, 66, cols, 1, false);
        ok &= check(f, 130, cols, 3, false);
        ok &= check(f, 128, cols, 70, true);
    }
    std::printf("metal_test: %s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
