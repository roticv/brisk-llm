// Throughput of the Metal Q4_0 matmul kernels on the shapes of Qwen3-1.7B.
#include <cstdio>

#include "metal/metal_session.h"

int main() {
    struct Shape {
        const char* name;
        std::size_t rows, cols;
    };
    const Shape shapes[] = {{"q/o 2048x2048", 2048, 2048}, {"gate/up 6144x2048", 6144, 2048}, {"down 2048x6144", 2048, 6144}};
    // Warm-up: the GPU clock ramps up over the first few milliseconds of work.
    brisk::metal_matmul_q4_0_seconds(6144, 2048, 512, true, 50);
    for (const Shape& s : shapes) {
        for (const std::size_t n : {std::size_t{1}, std::size_t{64}, std::size_t{128}, std::size_t{512}}) {
            const bool tiled = n >= 8;
            const double secs = brisk::metal_matmul_q4_0_seconds(s.rows, s.cols, n, tiled, 20);
            const double gflops = 2.0 * s.rows * s.cols * n / secs / 1e9;
            const double gbs = (s.rows * s.cols / 32.0 * 18) / secs / 1e9;
            std::printf("%-18s n %4zu %s: %7.3f ms  %7.0f GFLOPS  %6.1f GB/s weights\n", s.name, n, tiled ? "tiled" : "gemv ",
                        secs * 1e3, gflops, gbs);
        }
    }
}
