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
    Gemma4,  // Gemma 3 plus per-layer embeddings, K/V sharing, wider global heads, no attention scaling
    BitNet,  // ternary weights, extra norms before the output projections, squared-ReLU feed-forward
    Phi3,    // Phi-3/4-mini: fused QKV and gate/up tensors, partial rotary with LongRoPE factors
    LFM2,    // Liquid LFM2: short-convolution layers (gated 3-tap causal conv) between attention layers
};

// Query heads per key/value head, at most (the attention tiles size their query arrays by it).
inline constexpr std::size_t kMaxHeadsPerKv = 16;

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
    // LongRoPE (Phi): the factors for contexts up to rope_original_context
    // and beyond it; a session picks by its context length. rope_dim is how
    // many of a head's dimensions rotate (the rest pass through), and the
    // rotated values are multiplied by rope_attn_factor.
    std::vector<float> rope_factors_short, rope_factors_long;
    std::uint32_t rope_original_context = 0;
    std::uint32_t rope_dim = 0;  // 0: head_dim
    float rope_attn_factor = 1.0f;
    std::uint32_t rope_dim_of() const { return rope_dim != 0 ? rope_dim : head_dim; }
    // Gemma 3: every layer whose index is not 5 mod 6 attends only to the
    // last `sliding_window` positions and uses `rope_base_local`; the global
    // layers use rope_base scaled by `rope_scale`. 0 when no layer slides.
    std::uint32_t sliding_window = 0;
    float rope_base_local = 10000.0f;
    float rope_scale = 1.0f;
    float embedding_scale = 1.0f;     // Gemma: embeddings are multiplied by sqrt(embedding_dim)
    float final_softcap = 0.0f;       // logits = c * tanh(logits / c) when non-zero
    bool gelu = false;                // GELU (tanh approximation) gating instead of SiLU

    // Gemma 4: which layers slide comes from the file; the local layers' head
    // size may differ from the global ones' (head_dim); the last layers share
    // the K/V cache of the last layer of their kind before kv_from_start;
    // the feed-forward width varies per layer; per-layer embeddings of
    // ple_dim modulate the residual after each layer.
    std::vector<std::uint8_t> sliding_pattern;  // 1 = sliding, per layer (empty: every layer but each sixth)
    std::vector<std::uint32_t> kv_head_counts;   // per-layer key/value heads (Gemma 4 12B); empty: kv_head_count everywhere
    std::uint32_t head_dim_local = 0;           // 0: same as head_dim
    std::uint32_t kv_from_start = 0;            // 0: every layer has its own K/V
    std::vector<std::uint32_t> ff_dims;         // empty: feed_forward_dim for all
    std::uint32_t ple_dim = 0;
    float attention_scale = 0.0f;               // 0: 1 / sqrt(head size)
    bool value_norm = false;                    // Gemma 4: V is RMS-normalised (no weight) before the cache

    bool layer_slides(std::size_t layer) const {
        if (!sliding_pattern.empty()) return sliding_pattern[layer] != 0;
        return sliding_window != 0 && layer % 6 != 5;
    }
    std::uint32_t kv_head_count_of(std::size_t layer) const {
        return kv_head_counts.empty() ? kv_head_count : kv_head_counts[layer];
    }
    // LFM2: a layer with no key/value heads is a short-convolution layer.
    bool layer_is_conv(std::size_t layer) const { return kv_head_count_of(layer) == 0; }
    std::uint32_t conv_taps = 0;  // LFM2's shortconv.l_cache: the conv kernel's width in tokens
    std::uint32_t head_dim_of(std::size_t layer) const {
        return head_dim_local != 0 && layer_slides(layer) ? head_dim_local : head_dim;
    }
    std::size_t kv_layer_of(std::size_t layer) const {
        if (kv_from_start == 0 || layer < kv_from_start) return layer;
        return kv_from_start - (layer_slides(layer) ? 2 : 1);
    }
    std::uint32_t ff_dim_of(std::size_t layer) const { return ff_dims.empty() ? feed_forward_dim : ff_dims[layer]; }
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
// The file's architecture, or a failure naming the supported ones.
Architecture architecture_of(const GgufFile& file);

class Model {
public:
    struct Layer {
        std::span<const float> attention_norm;
        // LFM2 short-convolution layers: in_proj gives [b | c | x] (3 x embd), the
        // conv kernel is `conv_taps` weights per channel (oldest first), out_proj follows.
        Matrix conv_in, conv_out;
        std::span<const float> conv_weight;
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
        // Gemma 4: per-layer embedding gate (ple_dim x embd), projection (embd x
        // ple_dim) and the norm of its result; the layer output's scalar.
        Matrix ple_gate;
        Matrix ple_proj;
        std::span<const float> ple_post_norm;
        float out_scale = 1.0f;
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
    // Gemma 4's per-layer embeddings: the table (vocab x ple_dim * layers), the
    // projection of the input embedding (ple_dim * layers x embd) and its norm (ple_dim).
    const Matrix& ple_token_embedding() const { return ple_token_embedding_; }
    const Matrix& ple_model_proj() const { return ple_model_proj_; }
    std::span<const float> ple_proj_norm() const { return ple_proj_norm_; }
    std::span<const float> output_norm() const { return output_norm_; }
    const std::vector<Layer>& layers() const { return layers_; }

    // The configuration alone, before any weight is loaded (for the memory estimate).
    static ModelConfig parse_config(const GgufFile& file);

private:
    std::span<const float> load(const GgufFile& file, std::string_view name, std::size_t expected_elements);

    Matrix load_matrix(const GgufFile& file, std::string_view name, std::size_t rows, std::size_t cols);
    // Rows [row0, row0 + rows) of a tensor of total_rows rows (a fused QKV or gate/up tensor).
    Matrix load_matrix_rows(const GgufFile& file, std::string_view name, std::size_t row0, std::size_t rows,
                            std::size_t total_rows, std::size_t cols);

    ModelConfig config_;
    Matrix token_embedding_;
    Matrix output_projection_;
    Matrix ple_token_embedding_;
    Matrix ple_model_proj_;
    std::span<const float> ple_proj_norm_;
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
    // The caches hold half-precision values: half the bytes of float, and the
    // attention kernels widen as they read (llama.cpp's default too).
    std::vector<std::vector<std::uint16_t>> key_cache_;
    std::vector<std::vector<std::uint16_t>> value_cache_;
    std::vector<std::vector<std::uint16_t>> conv_cache_;  // LFM2: per conv layer, b*x per position (the conv's state)
    std::vector<float> conv_;                     // [token][3 * embd]: the in_proj output
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
    std::vector<float> rope_factors_;  // the RoPE angle divisors this session uses (see ModelConfig)
    std::vector<float> rope_;        // cos, sin pairs per position in the batch
    std::vector<float> rope_local_;  // the same with the sliding-window layers' base (Gemma)
    std::vector<float> ple_;         // Gemma 4: per-layer embedding inputs, [token][layer][ple_dim]
    std::vector<float> ple_gate_;    // and the gated values of one layer, [token][ple_dim]
    std::vector<float> logits_;
    std::vector<BlockQ8_0> q8_0_;  // activations quantised for the 32-block formats
    std::vector<BlockQ8_K> q8_k_;  // and for the K-quants
    std::vector<std::byte> prepared_[2];  // activations rearranged for a batched kernel, per kernel set
};

}  // namespace brisk
