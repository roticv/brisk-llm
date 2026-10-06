// Tries several ways of streaming memory to find the access pattern that gets
// closest to the hardware's read bandwidth. AArch64 only.
//
// Build: c++ -O3 -std=c++20 bench/membw_variants.cpp -o build/membw_variants
// Usage: membw_variants [mb]

#include <arm_neon.h>
#include <sys/mman.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

using Clock = std::chrono::steady_clock;
constexpr int kReps = 5;

__attribute__((noinline)) std::uint64_t scalar_sum(const std::uint8_t* p, std::size_t n) {
    const auto* q = reinterpret_cast<const std::uint64_t*>(p);
    std::uint64_t a = 0, b = 0, c = 0, d = 0;
    for (std::size_t i = 0; i + 4 <= n / 8; i += 4) {
        a += q[i];
        b += q[i + 1];
        c += q[i + 2];
        d += q[i + 3];
    }
    return a + b + c + d;
}

template <int kPrefetchDistance, bool kNonTemporal>
__attribute__((noinline)) std::uint64_t neon_sum(const std::uint8_t* p, std::size_t n) {
    uint64x2_t acc0 = vdupq_n_u64(0), acc1 = vdupq_n_u64(0), acc2 = vdupq_n_u64(0), acc3 = vdupq_n_u64(0);
    for (std::size_t i = 0; i + 64 <= n; i += 64) {
        if constexpr (kPrefetchDistance > 0) __builtin_prefetch(p + i + kPrefetchDistance);
        uint64x2_t v0, v1, v2, v3;
        if constexpr (kNonTemporal) {
            // ldnp: non-temporal pair load, a hint not to keep the lines in cache
            asm volatile("ldnp %q0, %q1, [%2]" : "=w"(v0), "=w"(v1) : "r"(p + i) : "memory");
            asm volatile("ldnp %q0, %q1, [%2]" : "=w"(v2), "=w"(v3) : "r"(p + i + 32) : "memory");
        } else {
            v0 = vld1q_u64(reinterpret_cast<const std::uint64_t*>(p + i));
            v1 = vld1q_u64(reinterpret_cast<const std::uint64_t*>(p + i + 16));
            v2 = vld1q_u64(reinterpret_cast<const std::uint64_t*>(p + i + 32));
            v3 = vld1q_u64(reinterpret_cast<const std::uint64_t*>(p + i + 48));
        }
        acc0 = vaddq_u64(acc0, v0);
        acc1 = vaddq_u64(acc1, v1);
        acc2 = vaddq_u64(acc2, v2);
        acc3 = vaddq_u64(acc3, v3);
    }
    const uint64x2_t acc = vaddq_u64(vaddq_u64(acc0, acc1), vaddq_u64(acc2, acc3));
    return vgetq_lane_u64(acc, 0) + vgetq_lane_u64(acc, 1);
}

using Fn = std::uint64_t (*)(const std::uint8_t*, std::size_t);

void run(const char* name, Fn fn, const std::uint8_t* p, std::size_t n, std::uint64_t expected) {
    double best = 0.0;
    for (int r = 0; r < kReps; ++r) {
        const auto start = Clock::now();
        const std::uint64_t got = fn(p, n);
        const std::chrono::duration<double> secs = Clock::now() - start;
        if (expected != 0 && got != expected) {
            std::fprintf(stderr, "%s: checksum mismatch\n", name);
            std::exit(1);
        }
        best = std::max(best, static_cast<double>(n) / secs.count() / 1e9);
    }
    std::printf("%-34s %6.2f GB/s\n", name, best);
}

std::uint8_t* allocate(std::size_t n, bool huge) {
    void* p = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
        std::perror("mmap");
        std::exit(1);
    }
#ifdef MADV_HUGEPAGE
    if (huge) madvise(p, n, MADV_HUGEPAGE);
#else
    (void)huge;
#endif
    std::memset(p, 1, n);
    return static_cast<std::uint8_t*>(p);
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t mb = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 256;
    const std::size_t n = mb << 20;

    std::uint8_t* p = allocate(n, false);
    const std::uint64_t expected = scalar_sum(p, n);
    run("scalar 64-bit sum", scalar_sum, p, n, expected);
    run("neon 4x16B", neon_sum<0, false>, p, n, expected);
    run("neon + prefetch 256B", neon_sum<256, false>, p, n, expected);
    run("neon + prefetch 512B", neon_sum<512, false>, p, n, expected);
    run("neon + prefetch 1024B", neon_sum<1024, false>, p, n, expected);
    run("neon + prefetch 2048B", neon_sum<2048, false>, p, n, expected);
    run("neon non-temporal", neon_sum<0, true>, p, n, expected);
    run("neon non-temporal + prefetch 512B", neon_sum<512, true>, p, n, expected);
    munmap(p, n);

    std::uint8_t* h = allocate(n, true);
    run("huge pages: neon 4x16B", neon_sum<0, false>, h, n, expected);
    run("huge pages: neon + prefetch 512B", neon_sum<512, false>, h, n, expected);
    munmap(h, n);
    return 0;
}
