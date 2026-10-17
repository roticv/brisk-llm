#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <span>
#include <string_view>
#include <vector>

#include "engine.h"
#include "gguf.h"
#include "quant.h"
#include "thread_pool.h"
#include "tokenizer.h"

namespace brisk {

// The model families the engine can run. They share the Llama-style layer
// structure and differ in a few extra norms and the feed-forward activation.
enum class Architecture {
    Qwen3,   // RMS norms on Q and K per head, SiLU-gated feed-forward
    Qwen2,   // Qwen2/2.5 and the DeepSeek-R1 distils of them: Q/K/V biases, no Q/K norms
    Llama,   // Llama 3.x, Mistral, SmolLM2 and other Llama-shaped models: adjacent-pair RoPE
    Gemma3,  // Q/K norms, norms after attention and the feed-forward, GELU gating, sliding-window layers
    BitNet,  // ternary weights, extra norms before the output projections, squared-ReLU feed-forward
};

struct ModelConfig {
    Architecture architecture = Architecture::Qwen3;
    std::uint32_t vocab_size = 0;
    std::uint32_t embedding_dim = 0;
    std::uint32_t layer_count = 0;
    std::uint32_t head_count = 0;     // query heads
    std::uint32_t kv_head_count = 0;  // key/value heads, shared between query heads
    std::uint32_t head_dim = 0;
    std::uint32_t feed_forward_dim = 0;
    std::uint32_t context_length = 0;  // longest sequence the model was trained on
    float rope_base = 0.0f;
    float rms_epsilon = 0.0f;
    // RoPE pairs element i with i + head_dim/2 ("NeoX": Qwen, BitNet) or with
    // i + 1 for even i ("normal": Llama).
    bool rope_neox = true;
    // Per-frequency divisors of the RoPE angles (Llama 3's long-context
    // scaling, from rope_freqs.weight); empty when the angles are unscaled.
    std::vector<float> rope_factors;
    // Gemma 3: every layer whose index is not 5 mod 6 attends only to the
    // last `sliding_window` positions and uses `rope_base_local`; the global
    // layers use rope_base scaled by `rope_scale`. 0 when no layer slides.
    std::uint32_t sliding_window = 0;
    float rope_base_local = 10000.0f;
    float rope_scale = 1.0f;
    float embedding_scale = 1.0f;     // Gemma: embeddings are multiplied by sqrt(embedding_dim)
    float final_softcap = 0.0f;       // logits = c * tanh(logits / c) when non-zero
    bool gelu = false;                // GELU (tanh approximation) gating instead of SiLU

    bool layer_slides(std::size_t layer) const { return sliding_window != 0 && layer % 6 != 5; }
};

// A weight matrix stored row by row: `rows` outputs, each over `cols` inputs.
// Quantised formats point at the file's bytes; F32 may too.
struct Matrix {
    WeightFormat format = WeightFormat::F32;
    std::span<const float> f32;        // when format is F32
    std::span<const std::byte> bytes;  // the blocks, otherwise
    std::size_t rows = 0;
    std::size_t cols = 0;

    std::size_t row_bytes() const { return cols / block_elements(format) * block_bytes(format); }
    std::span<const float> f32_row(std::size_t r) const { return f32.subspan(r * cols, cols); }
    // Start of row r's data; for Q4_0x4, r must be a multiple of 4 (rows are grouped).
    const std::byte* row(std::size_t r) const { return bytes.data() + r * row_bytes(); }
};

// Weights of a Qwen3 model. Float32 and Q4_0 tensors are used directly from
// the file; 16-bit float tensors are widened into memory owned by the model.
class Model {
public:
    struct Layer {
        std::span<const float> attention_norm;
        Matrix query;
        Matrix key;
        Matrix value;
        std::span<const float> query_norm;         // Qwen3: per-head norm of Q (empty otherwise)
        std::span<const float> key_norm;           // Qwen3: per-head norm of K
        std::span<const float> query_bias;         // Qwen2: biases added to the projections (empty otherwise)
        std::span<const float> key_bias;
        std::span<const float> value_bias;
        std::span<const float> attention_sub_norm;  // BitNet: norm of the attention result before the output projection
        std::span<const float> attention_post_norm;  // Gemma: norm of the projected attention before the residual add
        std::span<const float> feed_forward_post_norm;  // Gemma: the same after the down projection
        Matrix attention_output;
        std::span<const float> feed_forward_norm;
        Matrix gate;
        Matrix up;
        std::span<const float> feed_forward_sub_norm;  // BitNet: norm of gate*up before the down projection
        Matrix down;
    };

    // The buffer `file` was parsed from must outlive the model. Throws
    // std::runtime_error if the architecture or a tensor type is unsupported.
    // With `repack`, quantised matrices are rearranged at load for the CPU
    // kernels where that helps; the Metal backend wants them as in the file.
    explicit Model(const GgufFile& file, bool repack = true);

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    const ModelConfig& config() const { return config_; }
    const Matrix& token_embedding() const { return token_embedding_; }
    const Matrix& output_projection() const { return output_projection_; }
    std::span<const float> output_norm() const { return output_norm_; }
    const std::vector<Layer>& layers() const { return layers_; }

private:
    std::span<const float> load(const GgufFile& file, std::string_view name, std::size_t expected_elements);
    Matrix load_matrix(const GgufFile& file, std::string_view name, std::size_t rows, std::size_t cols);

    ModelConfig config_;
    Matrix token_embedding_;
    Matrix output_projection_;
    std::span<const float> output_norm_;
    std::vector<Layer> layers_;
    bool repack_ = true;
    std::deque<std::vector<float>> owned_;  // widened tensors; deque keeps them at stable addresses
    std::deque<std::vector<std::byte>> packed_;  // repacked quantised tensors
};

// One sequence being generated on the CPU: the key/value cache plus scratch
// space. The model must outlive the session.
class Session : public Engine {
public:
    // `threads` is the number of CPU threads to use, the caller included.
    // `max_context` bounds the sequence length (and the cache memory); 0 means
    // the model's trained context length.
    Session(const Model& model, std::size_t threads, std::size_t max_context = 0);

    // Appends tokens to the sequence and returns the logits for the token
    // after the last one. Several tokens at once (a prompt) are processed in
    // batches, which is much faster than one at a time. The span is valid
    // until the next call. Throws std::runtime_error past the model's context
    // length.
    using Engine::eval;
    std::span<const float> eval(std::span<const Token> tokens) override;
    std::span<const float> eval_all(std::span<const Token> tokens) override;
    void rollback(std::size_t position) override;
    std::size_t position() const override { return position_; }

    // Seconds spent in each phase of the forward pass so far, for profiling.
    struct Profile {
        double matmul = 0, quantize = 0, attention = 0, norm_rope = 0, activation = 0;
    };
    const Profile& profile() const { return profile_; }

private:
    // Processes up to kMaxBatch tokens starting at position_. With
    // `all_logits` set, logits_ receives one row per token instead of only
    // the last token's.
    void eval_batch(std::span<const Token> tokens, bool all_logits);
    // out = x * m^T for `n` rows of x (stride m.cols); out has stride m.rows.
    void multiply(const Matrix& m, const float* x, std::size_t n, float* out);
    // Several matrices applied to the same x in one pass (one activation
    // quantisation, one pool dispatch). All must have the same cols.
    struct Target {
        const Matrix* matrix;
        float* out;
    };
    void multiply(std::span<const Target> targets, const float* x, std::size_t n);

    const Model& model_;
    ThreadPool pool_;
    std::size_t position_ = 0;
    Profile profile_;

    // Per layer, head-major: [kv head][position][head_dim], so a head's rows
    // are contiguous and stream well. Allocated for max_context_ positions on
    // a layer's first use.
    std::size_t max_context_ = 0;
    std::vector<std::vector<float>> key_cache_;
    std::vector<std::vector<float>> value_cache_;
    std::vector<float> kv_staging_;  // this batch's new keys or values, [token][kv_dim], before copying in

    // Scratch, with one row per token of the current batch.
    std::vector<float> hidden_;
    std::vector<float> normed_;
    std::vector<float> query_;
    std::vector<float> attended_;
    std::vector<float> projected_;
    std::vector<float> gate_;
    std::vector<float> up_;
    std::vector<float> scores_;
    std::vector<float> rope_;        // cos, sin pairs per position in the batch
    std::vector<float> rope_local_;  // the same with the sliding-window layers' base (Gemma)
    std::vector<float> logits_;
    std::vector<BlockQ8_0> q8_0_;  // activations quantised for the 32-block formats
    std::vector<BlockQ8_K> q8_k_;  // and for the K-quants
    std::vector<std::byte> prepared_[2];  // activations rearranged for a batched kernel, per kernel set
};

}  // namespace brisk
