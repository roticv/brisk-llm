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

// The same for `n` activation rows at once: out[t * rows + r] = dot(row r of w, x row t).
// Each weight block is unpacked once and reused for every activation row, so
// this is compute-bound where the single-row version is memory-bound.
using Matmul = void (*)(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out);

struct KernelSet {
    Matvec matvec;
    Matmul matmul;
};

// The best kernels for `format` on this CPU. Throws std::runtime_error for F32.
KernelSet kernels_for(WeightFormat format);
std::string_view kernel_set_name();  // "generic", "neon" or "dotprod"

// Implementations, for tests and benchmarks. Generic ones exist for every
// format on every platform; NEON ones only where listed.
KernelSet generic_kernels(WeightFormat format);
#if defined(__aarch64__)
void matvec_q4_0_neon(const void* w, std::size_t rows, std::size_t cols, const void* x, float* out);
void matmul_q4_0_neon(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out);
void matvec_q4_0_dotprod(const void* w, std::size_t rows, std::size_t cols, const void* x, float* out);
void matmul_q4_0_dotprod(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out);
void matvec_q4_k_neon(const void* w, std::size_t rows, std::size_t cols, const void* x, float* out);
void matmul_q4_k_neon(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out);
void matvec_q4_k_dotprod(const void* w, std::size_t rows, std::size_t cols, const void* x, float* out);
void matmul_q4_k_dotprod(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out);
void matvec_q6_k_neon(const void* w, std::size_t rows, std::size_t cols, const void* x, float* out);
void matmul_q6_k_neon(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out);
void matvec_q6_k_dotprod(const void* w, std::size_t rows, std::size_t cols, const void* x, float* out);
void matmul_q6_k_dotprod(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out);
// Every NEON kernel set for `format`, best last; empty where only the generic exists.
struct NamedKernels {
    std::string_view name;
    KernelSet kernels;
};
std::vector<NamedKernels> neon_kernels(WeightFormat format);
#endif

}  // namespace brisk::kernels
