// Measures sequential read bandwidth, which bounds token generation speed:
// every weight is read once per token, so tokens/s <= bandwidth / model bytes.
//
// Build: c++ -O3 -std=c++20 -pthread bench/membw.cpp -o build/membw
// Usage: membw [max_threads] [mb_per_thread]

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kReps = 8;

// Sums the buffer as 64-bit words in four independent chains, so the loop is
// limited by memory rather than by the add dependency chain.
std::uint64_t read_pass(const std::uint64_t* p, std::size_t words) {
    std::uint64_t a = 0, b = 0, c = 0, d = 0;
    std::size_t i = 0;
    for (; i + 4 <= words; i += 4) {
        a += p[i];
        b += p[i + 1];
        c += p[i + 2];
        d += p[i + 3];
    }
    for (; i < words; ++i) a += p[i];
    return a + b + c + d;
}

double measure_gbps(int threads, std::size_t bytes_per_thread) {
    const std::size_t words = bytes_per_thread / sizeof(std::uint64_t);
    std::vector<std::vector<std::uint64_t>> bufs(static_cast<std::size_t>(threads));
    for (auto& buf : bufs) buf.assign(words, 1);  // touches every page

    double best = 0.0;
    for (int rep = 0; rep < kReps; ++rep) {
        std::atomic<int> ready{0};
        std::atomic<bool> go{false};
        std::atomic<std::uint64_t> sink{0};
        std::vector<std::thread> pool;
        for (int t = 0; t < threads; ++t) {
            pool.emplace_back([&, t] {
                ready.fetch_add(1);
                while (!go.load(std::memory_order_acquire)) {}
                sink.fetch_add(read_pass(bufs[static_cast<std::size_t>(t)].data(), words));
            });
        }
        while (ready.load() < threads) {}
        const auto start = Clock::now();
        go.store(true, std::memory_order_release);
        for (auto& th : pool) th.join();
        const std::chrono::duration<double> secs = Clock::now() - start;

        if (sink.load() != static_cast<std::uint64_t>(threads) * words) {
            std::fprintf(stderr, "checksum mismatch\n");
            std::exit(1);
        }
        const double gbps = static_cast<double>(threads) * bytes_per_thread / secs.count() / 1e9;
        best = std::max(best, gbps);
    }
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    const int hw = static_cast<int>(std::thread::hardware_concurrency());
    const int max_threads = argc > 1 ? std::atoi(argv[1]) : (hw > 0 ? hw : 4);
    const std::size_t mb = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 128;
    if (max_threads <= 0 || mb == 0) {
        std::fprintf(stderr, "usage: membw [max_threads] [mb_per_thread]\n");
        return 2;
    }

    std::printf("threads,mb_per_thread,read_gb_per_s\n");
    for (int t = 1; t <= max_threads; ++t) {
        std::printf("%d,%zu,%.2f\n", t, mb, measure_gbps(t, mb * 1024 * 1024));
        std::fflush(stdout);
    }
    return 0;
}
