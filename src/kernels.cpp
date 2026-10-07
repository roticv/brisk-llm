#include "kernels.h"

#include <stdexcept>

#include "cpu_features.h"

namespace brisk::kernels {

namespace {

template <typename Block, typename Activation, float (*Dot)(const Block*, const Activation*, std::size_t)>
void matvec_generic(const void* w, std::size_t rows, std::size_t cols, const void* x, float* out) {
    const auto* weights = static_cast<const Block*>(w);
    const auto* activations = static_cast<const Activation*>(x);
    const std::size_t blocks = cols / block_elements_of<Block>();
    for (std::size_t r = 0; r < rows; ++r) out[r] = Dot(weights + r * blocks, activations, blocks);
}

template <typename Block, typename Activation, float (*Dot)(const Block*, const Activation*, std::size_t)>
void matmul_generic(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out) {
    const auto* weights = static_cast<const Block*>(w);
    const auto* activations = static_cast<const Activation*>(x);
    const std::size_t blocks = cols / block_elements_of<Block>();
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t t = 0; t < n; ++t) out[t * rows + r] = Dot(weights + r * blocks, activations + t * blocks, blocks);
    }
}

void matvec_q4_0x4_generic(const void* w, std::size_t rows, std::size_t cols, const void* x, float* out) {
    const auto* weights = static_cast<const BlockQ4_0x4*>(w);
    const auto* activations = static_cast<const BlockQ8_0*>(x);
    const std::size_t blocks = cols / kBlockSize;
    for (std::size_t g = 0; g < rows / 4; ++g) dot_q4_0x4_q8_0(weights + g * blocks, activations, blocks, out + 4 * g);
}

void matmul_q4_0x4_generic(const void* w, std::size_t rows, std::size_t cols, const void* x, std::size_t n, float* out) {
    const auto* weights = static_cast<const BlockQ4_0x4*>(w);
    const auto* activations = static_cast<const BlockQ8_0*>(x);
    const std::size_t blocks = cols / kBlockSize;
    for (std::size_t g = 0; g < rows / 4; ++g) {
        for (std::size_t t = 0; t < n; ++t) {
            dot_q4_0x4_q8_0(weights + g * blocks, activations + t * blocks, blocks, out + t * rows + 4 * g);
        }
    }
}

}  // namespace

KernelSet generic_kernels(WeightFormat format) {
    switch (format) {
        case WeightFormat::Q4_0x4:
            return {matvec_q4_0x4_generic, matmul_q4_0x4_generic};
        case WeightFormat::Q4_0:
            return {matvec_generic<BlockQ4_0, BlockQ8_0, dot_q4_0_q8_0>, matmul_generic<BlockQ4_0, BlockQ8_0, dot_q4_0_q8_0>};
        case WeightFormat::Q4_1:
            return {matvec_generic<BlockQ4_1, BlockQ8_0, dot_q4_1_q8_0>, matmul_generic<BlockQ4_1, BlockQ8_0, dot_q4_1_q8_0>};
        case WeightFormat::Q4_K:
            return {matvec_generic<BlockQ4_K, BlockQ8_K, dot_q4_k_q8_k>, matmul_generic<BlockQ4_K, BlockQ8_K, dot_q4_k_q8_k>};
        case WeightFormat::Q6_K:
            return {matvec_generic<BlockQ6_K, BlockQ8_K, dot_q6_k_q8_k>, matmul_generic<BlockQ6_K, BlockQ8_K, dot_q6_k_q8_k>};
        case WeightFormat::F32:
            break;
    }
    throw std::runtime_error("kernels: no quantised kernels for F32 weights");
}

#if defined(__aarch64__)
std::vector<NamedKernels> neon_kernels(WeightFormat format) {
    std::vector<NamedKernels> out;
    const bool dotprod = cpu_features().dotprod;
    switch (format) {
        case WeightFormat::Q4_0:
            out.push_back({"neon", {matvec_q4_0_neon, matmul_q4_0_neon}});
            if (dotprod) out.push_back({"dotprod", {matvec_q4_0_dotprod, matmul_q4_0_dotprod}});
            if (dotprod && cpu_features().i8mm) out.push_back({"i8mm", {matvec_q4_0_dotprod, matmul_q4_0_i8mm}});
            break;
        case WeightFormat::Q4_K:
            out.push_back({"neon", {matvec_q4_k_neon, matmul_q4_k_neon}});
            if (dotprod) out.push_back({"dotprod", {matvec_q4_k_dotprod, matmul_q4_k_dotprod}});
            break;
        case WeightFormat::Q6_K:
            out.push_back({"neon", {matvec_q6_k_neon, matmul_q6_k_neon}});
            if (dotprod) out.push_back({"dotprod", {matvec_q6_k_dotprod, matmul_q6_k_dotprod}});
            break;
        case WeightFormat::Q4_0x4:
            if (dotprod && cpu_features().i8mm) {
                out.push_back({"i8mm", {matvec_q4_0x4_dotprod, matmul_q4_0x4_i8mm, prepare_q4_0x4_i8mm,
                                        prepared_bytes_q4_0x4_i8mm}});
            }
            break;
        case WeightFormat::Q4_1:
        case WeightFormat::F32:
            break;
    }
    return out;
}
#endif

bool prefers_q4_0x4() {
#if defined(__aarch64__)
    return cpu_features().dotprod && cpu_features().i8mm;
#else
    return false;
#endif
}

KernelSet kernels_for(WeightFormat format) {
#if defined(__aarch64__)
    const std::vector<NamedKernels> neon = neon_kernels(format);
    if (!neon.empty()) return neon.back().kernels;
#endif
    return generic_kernels(format);
}

std::string_view kernel_set_name() {
#if defined(__aarch64__)
    if (cpu_features().dotprod && cpu_features().i8mm) return "i8mm";
    return cpu_features().dotprod ? "dotprod" : "neon";
#else
    return "generic";
#endif
}

}  // namespace brisk::kernels
