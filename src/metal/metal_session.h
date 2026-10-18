#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include "engine.h"
#include "model.h"
#include "quant.h"

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

// For tests: multiplies a quantised matrix (rows x cols, GGUF blocks of
// `format`: Q4_0, Q4_1, Q4_K or Q6_K) by n rows of activations on the GPU,
// returning out[t][row]. `tiled` selects the 64x64 simdgroup-matrix kernel;
// otherwise the gemv kernel is used.
// With `paired` (gemv only), the matrix stands in for both gate and up and
// the result is silu(w x) * (w x), exercising the paired epilogue.
std::vector<float> metal_matmul(WeightFormat format, std::span<const std::byte> weights, std::size_t rows,
                                std::size_t cols, std::span<const float> x, std::size_t n, bool tiled,
                                bool paired = false);

// For benchmarks: GPU seconds per gemv of a rows x cols TQ2_0 matrix by one
// activation row, over `iterations` back-to-back dispatches. `variant` 0 is
// the production kernel; others are experiments compiled from the same source.
double metal_gemv_tq2_0_seconds(std::size_t rows, std::size_t cols, int iterations, int variant = 0);

// For benchmarks: GPU seconds per multiply of a rows x cols Q4_0 matrix by n
// activation rows, over `iterations` back-to-back dispatches of one command buffer.
double metal_matmul_q4_0_seconds(std::size_t rows, std::size_t cols, std::size_t n, bool tiled, int iterations);
// The same for any format the standalone matmul takes (Q4_0, Q4_1, Q4_K, Q6_K).
double metal_matmul_seconds(WeightFormat format, std::size_t rows, std::size_t cols, std::size_t n, bool tiled,
                            int iterations);

}  // namespace brisk
