// Checks the Metal Q4_0 matmul kernels against the CPU reference kernels.
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

bool check(std::size_t rows, std::size_t cols, std::size_t n, bool tiled) {
    std::mt19937 rng(static_cast<unsigned>(rows * 31 + cols * 7 + n));
    std::normal_distribution<float> dist;
    std::vector<float> x(n * cols);
    for (float& v : x) v = dist(rng);
    // Random blocks: a scale around 0.1 and uniformly random nibbles.
    std::vector<BlockQ4_0> blocks(rows * cols / 32);
    std::uniform_int_distribution<int> nibble(0, 255);
    for (BlockQ4_0& b : blocks) {
        const std::uint16_t d = float_to_half(0.05f + 0.1f * std::uniform_real_distribution<float>()(rng));
        std::memcpy(&b, &d, 2);
        for (std::uint8_t& q : b.qs) q = static_cast<std::uint8_t>(nibble(rng));
    }
    const std::span<const std::byte> bytes(reinterpret_cast<const std::byte*>(blocks.data()),
                                           blocks.size() * sizeof(BlockQ4_0));

    // Reference: the dequantised weights against the exact float activations.
    std::vector<float> w(rows * cols);
    dequantize_q4_0(blocks.data(), w.data(), w.size());
    std::vector<float> expected(rows * n);
    for (std::size_t t = 0; t < n; ++t) {
        for (std::size_t r = 0; r < rows; ++r) {
            double sum = 0.0;
            for (std::size_t c = 0; c < cols; ++c) sum += double(w[r * cols + c]) * x[t * cols + c];
            expected[t * rows + r] = static_cast<float>(sum);
        }
    }
    const std::vector<float> got = metal_matmul_q4_0(bytes, rows, cols, x, n, tiled);

    if (std::getenv("METAL_TEST_DUMP") != nullptr) {
        for (std::size_t i = 0; i < std::min<std::size_t>(8, expected.size()); ++i) std::printf("  [%zu] expected %g got %g\n", i, expected[i], got[i]);
    }
    double worst = 0.0;
    for (std::size_t i = 0; i < expected.size(); ++i) worst = std::max(worst, std::fabs(double(expected[i]) - got[i]));
    // Half activations in the tiled kernel: relative error ~1e-3 of a dot of `cols` unit terms.
    const double tolerance = (tiled ? 4e-3 : 1e-4) * std::sqrt(double(cols));
    const bool ok = worst <= tolerance;
    std::printf("%s rows %zu cols %zu n %zu: max abs error %.3g (tolerance %.3g) %s\n", tiled ? "tiled" : "gemv ", rows,
                cols, n, worst, tolerance, ok ? "ok" : "FAILED");
    return ok;
}

}  // namespace

int main() {
    if (!MetalSession::available()) {
        std::printf("metal_test: no Metal device, skipped\n");
        return 0;
    }
    bool ok = true;
    ok &= check(64, 64, 1, false);
    ok &= check(130, 2048, 3, false);
    ok &= check(1024, 2048, 7, false);
    ok &= check(128, 64, 8, true);
    ok &= check(128, 2048, 64, true);
    ok &= check(256, 1024, 100, true);
    ok &= check(384, 96, 130, true);
    std::printf("metal_test: %s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
