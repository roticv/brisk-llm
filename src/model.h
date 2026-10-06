#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <span>
#include <string_view>
#include <vector>

#include "gguf.h"
#include "tokenizer.h"

namespace brisk {

struct ModelConfig {
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
};

// A weight matrix stored row by row: `rows` outputs, each over `cols` inputs.
struct Matrix {
    std::span<const float> data;
    std::size_t rows = 0;
    std::size_t cols = 0;

    std::span<const float> row(std::size_t r) const { return data.subspan(r * cols, cols); }
};

// Weights of a Qwen3 model as float32. Float32 tensors are used directly from
// the file; 16-bit float tensors are widened into memory owned by the model.
class Model {
public:
    struct Layer {
        std::span<const float> attention_norm;
        Matrix query;
        Matrix key;
        Matrix value;
        std::span<const float> query_norm;
        std::span<const float> key_norm;
        Matrix attention_output;
        std::span<const float> feed_forward_norm;
        Matrix gate;
        Matrix up;
        Matrix down;
    };

    // The buffer `file` was parsed from must outlive the model. Throws
    // std::runtime_error if the architecture or a tensor type is unsupported.
    explicit Model(const GgufFile& file);

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
    std::deque<std::vector<float>> owned_;  // widened tensors; deque keeps them at stable addresses
};

// One sequence being generated: the key/value cache plus scratch space.
// The model must outlive the session.
class Session {
public:
    explicit Session(const Model& model);

    // Feeds the next token of the sequence and returns the logits for the
    // token after it. The span is valid until the next call. Throws
    // std::runtime_error past the model's context length.
    std::span<const float> eval(Token token);

    std::size_t position() const { return position_; }

private:
    const Model& model_;
    std::size_t position_ = 0;

    // Per layer, one entry per position: kv_head_count * head_dim floats.
    std::vector<std::vector<float>> key_cache_;
    std::vector<std::vector<float>> value_cache_;

    std::vector<float> hidden_;
    std::vector<float> normed_;
    std::vector<float> query_;
    std::vector<float> attended_;
    std::vector<float> projected_;
    std::vector<float> gate_;
    std::vector<float> up_;
    std::vector<float> scores_;
    std::vector<float> rope_;  // cos, sin pairs for the current position
    std::vector<float> logits_;
};

}  // namespace brisk
