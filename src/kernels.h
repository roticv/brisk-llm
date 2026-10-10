#pragma once

#include <cstddef>
#include <string_view>
#include <vector>

#include "quant.h"

// The hot loops, with one implementation per instruction set. The best
// implementation the CPU supports is chosen the first time it is used.
//
// Weights are `rows` rows of `cols` elements in a quantised format; `cols`
// must be a multiple of the format's block size. Activations are already
// quantised to the format the weights take (see takes_q8_k).

namespace brisk::kernels {

// out[r] = dot(row r of w, x).
using Matvec = void (*)(const void* w, std::size_t rows, std::size_t cols, const void* x, float* out);

// The same for `n` activation rows at once: out[t * out_stride + r] = dot(row r of w, x row t).
// Each weight block is unpacked once and reused for every activation row, so
// this is compute-bound where the single-row version is memory-bound.
using Matmul = void (*)(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out,
                        std::size_t out_stride);

// Some batched kernels want the activations rearranged first. `prepare`
// writes `prepared_bytes(n, blocks)` bytes that `matmul` then takes as `x`
// in place of the raw activation rows; both are null when not needed.
using Prepare = void (*)(const void* x, std::size_t n, std::size_t blocks, void* out);
using PreparedBytes = std::size_t (*)(std::size_t n, std::size_t blocks);

struct KernelSet {
    Matvec matvec;
    Matmul matmul;
    Prepare prepare = nullptr;
    PreparedBytes prepared_bytes = nullptr;
};

// The best kernels for `format` on this CPU. Throws std::runtime_error for F32.
KernelSet kernels_for(WeightFormat format);
std::string_view kernel_set_name();  // "generic", "neon", "dotprod" or "i8mm"
// True when Q4_0 (TQ2_0) matrices should be repacked to the four-row
// interleaved layout at load for this CPU.
bool prefers_q4_0x4();

// Implementations, for tests and benchmarks. Generic ones exist for every
// format on every platform; NEON ones only where listed.
KernelSet generic_kernels(WeightFormat format);
#if defined(__aarch64__)
void matvec_q4_0_neon(const void* w, std::size_t rows, std::size_t cols, const void* x, float* out);
void matmul_q4_0_neon(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out,
                      std::size_t out_stride);
void matvec_q4_0_dotprod(const void* w, std::size_t rows, std::size_t cols, const void* x, float* out);
void matmul_q4_0_dotprod(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out,
                      std::size_t out_stride);
void matmul_q4_0_i8mm(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out,
                      std::size_t out_stride);
void matvec_q4_kx4_dotprod(const void* w, std::size_t rows, std::size_t cols, const void* x, float* out);
void matmul_q4_kx4_i8mm(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out,
                      std::size_t out_stride);
void matvec_q6_kx4_dotprod(const void* w, std::size_t rows, std::size_t cols, const void* x, float* out);
void matmul_q6_kx4_i8mm(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out,
                      std::size_t out_stride);
// Activation preparation shared by the smmla tiles of all 256-element formats.
void prepare_k_i8mm(const void* x, std::size_t n, std::size_t blocks, void* out);
std::size_t prepared_bytes_k_i8mm(std::size_t n, std::size_t blocks);
void matvec_tq2_0x4_dotprod(const void* w, std::size_t rows, std::size_t cols, const void* x, float* out);
void matmul_tq2_0x4_i8mm(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out,
                      std::size_t out_stride);
void matvec_q4_0x4_dotprod(const void* w, std::size_t rows, std::size_t cols, const void* x, float* out);
void matmul_q4_0x4_i8mm(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out,
                      std::size_t out_stride);
void prepare_q4_0x4_i8mm(const void* x, std::size_t n, std::size_t blocks, void* out);
std::size_t prepared_bytes_q4_0x4_i8mm(std::size_t n, std::size_t blocks);
void matvec_q4_1_neon(const void* w, std::size_t rows, std::size_t cols, const void* x, float* out);
void matmul_q4_1_neon(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out,
                      std::size_t out_stride);
void matvec_q4_1_dotprod(const void* w, std::size_t rows, std::size_t cols, const void* x, float* out);
void matmul_q4_1_dotprod(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out,
                      std::size_t out_stride);
void matvec_q4_k_neon(const void* w, std::size_t rows, std::size_t cols, const void* x, float* out);
void matmul_q4_k_neon(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out,
                      std::size_t out_stride);
void matvec_q4_k_dotprod(const void* w, std::size_t rows, std::size_t cols, const void* x, float* out);
void matmul_q4_k_dotprod(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out,
                      std::size_t out_stride);
void matvec_tq2_0_neon(const void* w, std::size_t rows, std::size_t cols, const void* x, float* out);
void matmul_tq2_0_neon(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out,
                      std::size_t out_stride);
void matvec_tq2_0_dotprod(const void* w, std::size_t rows, std::size_t cols, const void* x, float* out);
void matmul_tq2_0_dotprod(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out,
                      std::size_t out_stride);
void matvec_q6_k_neon(const void* w, std::size_t rows, std::size_t cols, const void* x, float* out);
void matmul_q6_k_neon(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out,
                      std::size_t out_stride);
void matvec_q6_k_dotprod(const void* w, std::size_t rows, std::size_t cols, const void* x, float* out);
void matmul_q6_k_dotprod(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out,
                      std::size_t out_stride);
// Every NEON kernel set for `format`, best last; empty where only the generic exists.
struct NamedKernels {
    std::string_view name;
    KernelSet kernels;
};
std::vector<NamedKernels> neon_kernels(WeightFormat format);
#endif

}  // namespace brisk::kernels
