// Times the attention kernels alone at prompt-sized shapes: scores for a
// tile of queries against all cached keys, then the weighted value sum.
// Built as part of the normal CMake build. Usage: attention_bench [positions]

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "kernels_f32.h"

int main(int argc, char** argv) {
    const std::size_t positions = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 497;
    const std::size_t dim = 128, kv_dim = dim, heads = 2, tile = 16;  // head-major cache: rows are contiguous
    std::mt19937 rng(1);
    std::normal_distribution<float> normal;
    std::vector<float> keys(positions * kv_dim), values(positions * kv_dim), queries(tile * heads * dim);
    for (float& v : keys) v = normal(rng);
    for (float& v : values) v = normal(rng);
    for (float& v : queries) v = normal(rng);
    std::vector<float> scores(tile * heads * positions), out(tile * heads * dim);
    std::vector<const float*> q(tile * heads);
    std::vector<float*> rows(tile * heads);
    for (std::size_t i = 0; i < q.size(); ++i) {
        q[i] = queries.data() + i * dim;
        rows[i] = scores.data() + i * positions;
    }

    using Clock = std::chrono::steady_clock;
    auto bench = [&](const char* name, double macs, auto fn) {
        int reps = 0;
        const auto start = Clock::now();
        while (std::chrono::duration<double>(Clock::now() - start).count() < 1.0) {
            fn();
            ++reps;
        }
        const double secs = std::chrono::duration<double>(Clock::now() - start).count();
        std::printf("%-14s %7.2f GMAC/s  (%zu positions, %zu queries)\n", name, macs * reps / secs / 1e9, positions, q.size());
    };
    const double macs = static_cast<double>(q.size()) * positions * dim;
    bench("scores", macs, [&] {
        brisk::kernels::scores_f32(q.data(), q.size(), keys.data(), kv_dim, positions, dim, 0.1f, rows.data());
    });
    bench("softmax", 0.0, [&] {
        for (std::size_t i = 0; i < q.size(); ++i) brisk::kernels::softmax_f32({rows[i], positions});
    });
    bench("weighted_sum", macs, [&] {
        for (std::size_t t = 0; t < tile; ++t) {
            brisk::kernels::weighted_sum_f32(values.data(), kv_dim, positions, rows[t * heads], positions, heads, dim,
                                             out.data() + t * heads * dim);
        }
    });
    return 0;
}
