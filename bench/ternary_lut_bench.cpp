// Prototype of a lookup-table kernel for ternary weights x int8 activations
// (the T-MAC / bitnet.cpp idea) against brisk's current baseline-NEON ternary
// kernel, one thread, batched over 64 tokens: multiply-adds per second and the
// largest deviation of the result.
//
// Layout for the lookup kernel: for every group of 16 rows and every pair of
// columns, one byte per row holding the pair's index w0 * 3 + w1 (codes
// 0..2); two indices per byte would halve the bytes but cost an unpack, so the
// prototype keeps one per byte. Per token and column pair, a 16-entry table of
// the 9 sums (-x0-x1 .. x0+x1) is built once and `tbl` reads 16 rows' partial
// sums with one instruction; nothing goes through the multiplier.
#include <arm_neon.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "kernels.h"
#include "quant.h"

namespace {

using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }

// Tables for one token: per column pair, the 9 sums (-x0-x1 .. x0+x1) at index
// w0 * 3 + w1 (w = code - 1), as a 16-byte table of low bytes and one of high
// bytes. Built once per token and shared by every row group.
void build_tables_i16(const std::int8_t* x, std::size_t pairs, std::uint8_t* tables /* [pairs][32] */) {
    for (std::size_t p = 0; p < pairs; ++p) {
        const int x0 = x[2 * p], x1 = x[2 * p + 1];
        std::int16_t table[16] = {};
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) table[a * 3 + b] = static_cast<std::int16_t>((a - 1) * x0 + (b - 1) * x1);
        }
        const int16x8_t t0 = vld1q_s16(table), t1 = vld1q_s16(table + 8);
        vst1q_s8(reinterpret_cast<std::int8_t*>(tables + 32 * p), vuzp1q_s8(vreinterpretq_s8_s16(t0), vreinterpretq_s8_s16(t1)));
        vst1q_s8(reinterpret_cast<std::int8_t*>(tables + 32 * p + 16), vuzp2q_s8(vreinterpretq_s8_s16(t0), vreinterpretq_s8_s16(t1)));
    }
}

void build_tables_i8(const std::int8_t* x, std::size_t pairs, std::uint8_t* tables /* [pairs][16] */) {
    for (std::size_t p = 0; p < pairs; ++p) {
        const int x0 = x[2 * p] >> 1, x1 = x[2 * p + 1] >> 1;
        std::int8_t table[16] = {};
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) table[a * 3 + b] = static_cast<std::int8_t>((a - 1) * x0 + (b - 1) * x1);
        }
        std::memcpy(tables + 16 * p, table, 16);
    }
}

// Exact: int16 tables as two byte tables (low and high bytes), accumulated in int16 lanes.
void lut_rows16_i16(const std::uint8_t* idx /* [pairs][16] */, std::size_t pairs, const std::uint8_t* tables, std::int32_t out[16]) {
    int32x4_t acc[4] = {vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0)};
    int16x8_t lo = vdupq_n_s16(0), hi = vdupq_n_s16(0);
    for (std::size_t p = 0; p < pairs; ++p) {
        const int8x16_t tlo = vld1q_s8(reinterpret_cast<const std::int8_t*>(tables + 32 * p));
        const int8x16_t thi = vld1q_s8(reinterpret_cast<const std::int8_t*>(tables + 32 * p + 16));
        const uint8x16_t i = vld1q_u8(idx + 16 * p);
        const int8x16_t vlo = vqtbl1q_s8(tlo, i), vhi = vqtbl1q_s8(thi, i);
        // Interleave low/high bytes back into int16 for rows 0..7 and 8..15.
        lo = vaddq_s16(lo, vreinterpretq_s16_s8(vzip1q_s8(vlo, vhi)));
        hi = vaddq_s16(hi, vreinterpretq_s16_s8(vzip2q_s8(vlo, vhi)));
        if ((p & 63) == 63) {  // widen before an int16 lane could overflow (64 pairs x 254)
            acc[0] = vaddw_s16(acc[0], vget_low_s16(lo));
            acc[1] = vaddw_s16(acc[1], vget_high_s16(lo));
            acc[2] = vaddw_s16(acc[2], vget_low_s16(hi));
            acc[3] = vaddw_s16(acc[3], vget_high_s16(hi));
            lo = vdupq_n_s16(0);
            hi = vdupq_n_s16(0);
        }
    }
    acc[0] = vaddw_s16(acc[0], vget_low_s16(lo));
    acc[1] = vaddw_s16(acc[1], vget_high_s16(lo));
    acc[2] = vaddw_s16(acc[2], vget_low_s16(hi));
    acc[3] = vaddw_s16(acc[3], vget_high_s16(hi));
    for (int k = 0; k < 4; ++k) vst1q_s32(out + 4 * k, acc[k]);
}

// Approximate: activations halved to 7 bits so a pair's sum fits int8; one table, widening adds.
void lut_rows16_i8(const std::uint8_t* idx, std::size_t pairs, const std::uint8_t* tables, std::int32_t out[16]) {
    int32x4_t acc[4] = {vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0)};
    int16x8_t lo = vdupq_n_s16(0), hi = vdupq_n_s16(0);
    for (std::size_t p = 0; p < pairs; ++p) {
        const int8x16_t t = vld1q_s8(reinterpret_cast<const std::int8_t*>(tables + 16 * p));
        const int8x16_t v = vqtbl1q_s8(t, vld1q_u8(idx + 16 * p));
        lo = vaddw_s8(lo, vget_low_s8(v));
        hi = vaddw_s8(hi, vget_high_s8(v));
        if ((p & 127) == 127) {
            acc[0] = vaddw_s16(acc[0], vget_low_s16(lo));
            acc[1] = vaddw_s16(acc[1], vget_high_s16(lo));
            acc[2] = vaddw_s16(acc[2], vget_low_s16(hi));
            acc[3] = vaddw_s16(acc[3], vget_high_s16(hi));
            lo = vdupq_n_s16(0);
            hi = vdupq_n_s16(0);
        }
    }
    acc[0] = vaddw_s16(acc[0], vget_low_s16(lo));
    acc[1] = vaddw_s16(acc[1], vget_high_s16(lo));
    acc[2] = vaddw_s16(acc[2], vget_low_s16(hi));
    acc[3] = vaddw_s16(acc[3], vget_high_s16(hi));
    for (int k = 0; k < 4; ++k) vst1q_s32(out + 4 * k, 2 * acc[k]);  // the halving undone
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t rows = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 2048;
    const std::size_t cols = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 2048;
    constexpr std::size_t kTokens = 64;
    const std::size_t blocks = cols / 256, pairs = cols / 2;
    std::mt19937 rng(1);

    // Random ternary weights as codes 0..2, in TQ2_0 blocks for brisk's kernel and as row-group indices for the lookup.
    std::vector<std::uint8_t> codes(rows * cols);
    for (std::uint8_t& c : codes) c = static_cast<std::uint8_t>(rng() % 3);
    std::vector<brisk::BlockTQ2_0> w(rows * blocks);
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t b = 0; b < blocks; ++b) {
            brisk::BlockTQ2_0& blk = w[r * blocks + b];
            std::memset(blk.qs, 0, sizeof(blk.qs));
            blk.d = brisk::float_to_half(0.05f);
            // TQ2_0: byte j + m (j in {0, 32}, m < 32) holds at bits 2l element j * 4 + 32 * l + m.
            for (int e = 0; e < 256; ++e) {
                const int j = (e / 128) * 32, within = e % 128, l = within / 32, m = within % 32;
                blk.qs[j + m] |= static_cast<std::uint8_t>(codes[r * cols + b * 256 + e] << (2 * l));
            }
        }
    }
    std::vector<std::uint8_t> idx(rows / 16 * pairs * 16);  // [group][pair][16 rows]
    for (std::size_t g = 0; g < rows / 16; ++g) {
        for (std::size_t p = 0; p < pairs; ++p) {
            for (int l = 0; l < 16; ++l) {
                const std::size_t r = g * 16 + l;
                idx[(g * pairs + p) * 16 + l] = static_cast<std::uint8_t>(codes[r * cols + 2 * p] * 3 + codes[r * cols + 2 * p + 1]);
            }
        }
    }

    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> x(kTokens * cols);
    for (float& v : x) v = normal(rng);
    std::vector<brisk::BlockQ8_K> xq(kTokens * blocks);
    for (std::size_t t = 0; t < kTokens; ++t) brisk::quantize_q8_k(x.data() + t * cols, xq.data() + t * blocks, cols);
    // The lookup kernel takes each token's int8 row contiguous (and the per-block scales separately).
    std::vector<std::int8_t> x8(kTokens * cols);
    for (std::size_t t = 0; t < kTokens; ++t) {
        for (std::size_t b = 0; b < blocks; ++b) std::memcpy(x8.data() + t * cols + b * 256, xq[t * blocks + b].qs, 256);
    }

    const double macs = static_cast<double>(rows) * cols * kTokens;
    std::printf("%zu x %zu ternary, %zu tokens, one thread\n", rows, cols, kTokens);

    // Reference: brisk's kernel (the "neon" set).
    std::vector<float> ref(kTokens * rows), out(kTokens * rows);
    brisk::kernels::KernelSet ks;
    for (const brisk::kernels::NamedKernels& nk : brisk::kernels::neon_kernels(brisk::WeightFormat::TQ2_0)) {
        if (nk.name == "neon") ks = nk.kernels;
    }
    {
        int reps = 0;
        const auto start = Clock::now();
        while (seconds(start) < 1.0) {
            ks.matmul(w.data(), rows, cols, xq.data(), kTokens, ref.data(), rows);
            ++reps;
        }
        std::printf("%-26s %6.2f GMAC/s\n", "brisk neon (vmull chain)", macs * reps / seconds(start) / 1e9);
    }

    // Lookup variants. Integer results per (token, row group) are scaled per 256-block... the prototype
    // uses one weight scale per row (all blocks share d here) and the token's per-block activation
    // scales are folded by scoring per block: to keep the prototype simple the activation scale is
    // applied per 256-column block by running the kernel per block.
    auto run_lut = [&](const char* name, auto build, std::size_t table_bytes, auto kernel) {
        std::vector<std::uint8_t> tables(pairs * table_bytes);
        int reps = 0;
        const auto start = Clock::now();
        while (seconds(start) < 1.0) {
            for (std::size_t t = 0; t < kTokens; ++t) {
                build(x8.data() + t * cols, pairs, tables.data());
                for (std::size_t g = 0; g < rows / 16; ++g) {
                    float sums[16] = {};
                    for (std::size_t b = 0; b < blocks; ++b) {
                        std::int32_t part[16];
                        kernel(idx.data() + (g * pairs + b * 128) * 16, 128, tables.data() + b * 128 * table_bytes, part);
                        const float scale = xq[t * blocks + b].d;
                        for (int l = 0; l < 16; ++l) sums[l] += scale * static_cast<float>(part[l]);
                    }
                    for (int l = 0; l < 16; ++l) out[t * rows + g * 16 + l] = sums[l] * 0.05f;
                }
            }
            ++reps;
        }
        double worst = 0.0, scale = 0.0;
        for (std::size_t i = 0; i < out.size(); ++i) {
            worst = std::max(worst, std::fabs(static_cast<double>(out[i]) - ref[i]));
            scale = std::max(scale, std::fabs(static_cast<double>(ref[i])));
        }
        std::printf("%-26s %6.2f GMAC/s   max |diff| %.4f of %.2f\n", name, macs * reps / seconds(start) / 1e9, worst, scale);
    };
    run_lut("lookup, int16 tables", build_tables_i16, 32, lut_rows16_i16);
    run_lut("lookup, int8 tables (7-bit x)", build_tables_i8, 16, lut_rows16_i8);

    // The engine's kernel on the same weights: the TQ2_0x16 layout with its prepare step (7-bit activations).
    {
        std::vector<brisk::BlockTQ2_0x16> packed(rows / 16 * blocks);
        brisk::repack_tq2_0x16(w.data(), rows, blocks, packed.data());
        std::vector<brisk::BlockQ8_K> xq7(kTokens * blocks);
        for (std::size_t t = 0; t < kTokens; ++t) brisk::quantize_q8_k(x.data() + t * cols, xq7.data() + t * blocks, cols, 7);
        brisk::kernels::KernelSet lk;
        for (const brisk::kernels::NamedKernels& nk : brisk::kernels::neon_kernels(brisk::WeightFormat::TQ2_0x16)) lk = nk.kernels;
        std::vector<std::uint8_t> prepared(lk.prepared_bytes(kTokens, blocks));
        int reps = 0;
        const auto start = Clock::now();
        while (seconds(start) < 1.0) {
            lk.prepare(xq7.data(), kTokens, blocks, prepared.data());
            lk.matmul(packed.data(), rows, cols, prepared.data(), kTokens, out.data(), rows);
            ++reps;
        }
        double worst = 0.0;
        for (std::size_t i = 0; i < out.size(); ++i) worst = std::max(worst, std::fabs(static_cast<double>(out[i]) - ref[i]));
        std::printf("%-26s %6.2f GMAC/s   max |diff| %.4f\n", "engine TQ2_0x16 + prepare", macs * reps / seconds(start) / 1e9, worst);
    }

    // One token (generation): weight bytes streamed per second by each matvec.
    {
        const double bytes = static_cast<double>(rows) * cols / 4;
        int reps = 0;
        auto start = Clock::now();
        while (seconds(start) < 1.0) {
            ks.matvec(w.data(), rows, cols, xq.data(), out.data());
            ++reps;
        }
        std::printf("%-26s %6.2f GB/s (matvec)\n", "brisk neon (vmull chain)", bytes * reps / seconds(start) / 1e9);
        std::vector<brisk::BlockTQ2_0x16> packed(rows / 16 * blocks);
        brisk::repack_tq2_0x16(w.data(), rows, blocks, packed.data());
        std::vector<brisk::BlockQ8_K> xq7(blocks);
        brisk::quantize_q8_k(x.data(), xq7.data(), cols, 7);
        brisk::kernels::KernelSet lk;
        for (const brisk::kernels::NamedKernels& nk : brisk::kernels::neon_kernels(brisk::WeightFormat::TQ2_0x16)) lk = nk.kernels;
        std::vector<std::uint8_t> prepared(lk.prepared_bytes(1, blocks));
        reps = 0;
        start = Clock::now();
        while (seconds(start) < 1.0) {
            lk.prepare(xq7.data(), 1, blocks, prepared.data());
            lk.matvec(packed.data(), rows, cols, prepared.data(), out.data());
            ++reps;
        }
        std::printf("%-26s %6.2f GB/s (matvec)\n", "engine TQ2_0x16 + prepare", bytes * reps / seconds(start) / 1e9);
    }
    return 0;
}
