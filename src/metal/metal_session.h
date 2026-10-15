#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include "engine.h"
#include "model.h"

namespace brisk {

// The forward pass on an Apple GPU. The model's quantised weights are used
// in place from the memory-mapped file (unified memory), the K/V cache lives
// on the GPU, and each batch of tokens is one command buffer.
//
// Supports Qwen3 and BitNet models whose matrices are Q4_0 in the file;
// throws std::runtime_error otherwise. `file` is the mapping the model was
// parsed from and must outlive the session.
class MetalSession : public Engine {
public:
    MetalSession(const Model& model, std::span<const std::byte> file, std::size_t max_context = 0);
    ~MetalSession() override;

    using Engine::eval;
    std::span<const float> eval(std::span<const Token> tokens) override;
    std::span<const float> eval_all(std::span<const Token> tokens) override;
    void rollback(std::size_t position) override;
    std::size_t position() const override;

    static bool available();  // a Metal device exists

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// For tests: multiplies a Q4_0 matrix (rows x cols, GGUF blocks) by n rows of
// activations on the GPU, returning out[t][row]. `tiled` selects the batched
// simdgroup-matrix kernel; otherwise the gemv kernel is used.
std::vector<float> metal_matmul_q4_0(std::span<const std::byte> weights, std::size_t rows, std::size_t cols,
                                     std::span<const float> x, std::size_t n, bool tiled);

// For benchmarks: GPU seconds per multiply of a rows x cols Q4_0 matrix by n
// activation rows, over `iterations` back-to-back dispatches of one command buffer.
double metal_matmul_q4_0_seconds(std::size_t rows, std::size_t cols, std::size_t n, bool tiled, int iterations);

}  // namespace brisk
