#include "model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace brisk {

namespace {

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("model: " + what); }

float half_to_float(std::uint16_t half) {
    const std::uint32_t h = half;
    const std::uint32_t sign = (h & 0x8000) << 16;
    std::uint32_t exponent = (h >> 10) & 0x1F;
    std::uint32_t mantissa = h & 0x3FF;
    std::uint32_t bits = 0;
    if (exponent == 0x1F) {
        bits = sign | 0x7F800000 | (mantissa << 13);  // infinity or NaN
    } else if (exponent != 0) {
        bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
    } else if (mantissa != 0) {
        // Subnormal half: shift the mantissa up until its leading bit is implicit.
        exponent = 113;
        while ((mantissa & 0x400) == 0) {
            mantissa <<= 1;
            --exponent;
        }
        bits = sign | (exponent << 23) | ((mantissa & 0x3FF) << 13);
    } else {
        bits = sign;
    }
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

float bfloat_to_float(std::uint16_t b) {
    const std::uint32_t bits = static_cast<std::uint32_t>(b) << 16;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

// Independent partial sums let the compiler vectorise the loop without
// needing permission to reorder floating-point additions.
float dot(const float* a, const float* b, std::size_t n) {
    constexpr std::size_t kLanes = 16;
    float lanes[kLanes] = {};
    std::size_t i = 0;
    for (; i + kLanes <= n; i += kLanes) {
        for (std::size_t j = 0; j < kLanes; ++j) lanes[j] += a[i + j] * b[i + j];
    }
    float sum = 0.0f;
    for (; i < n; ++i) sum += a[i] * b[i];
    for (const float lane : lanes) sum += lane;
    return sum;
}

void multiply(const Matrix& m, std::span<const float> x, std::span<float> out) {
    for (std::size_t r = 0; r < m.rows; ++r) out[r] = dot(m.data.data() + r * m.cols, x.data(), m.cols);
}

// out = x / sqrt(mean(x^2) + epsilon) * weight. `x` and `out` may alias.
void rms_norm(std::span<const float> x, std::span<const float> weight, float epsilon, std::span<float> out) {
    double sum = 0.0;
    for (const float v : x) sum += static_cast<double>(v * v);  // squared in float, as llama.cpp does
    const float mean = static_cast<float>(sum / static_cast<double>(x.size()));
    const float scale = 1.0f / std::sqrt(mean + epsilon);
    for (std::size_t i = 0; i < x.size(); ++i) out[i] = x[i] * scale * weight[i];
}

// Rotary position embedding in the "NeoX" layout: element i is paired with
// element i + dim/2. `rope` holds cos, sin for each pair.
void apply_rope(std::span<float> head, std::span<const float> rope) {
    const std::size_t half = head.size() / 2;
    for (std::size_t i = 0; i < half; ++i) {
        const float c = rope[2 * i];
        const float s = rope[2 * i + 1];
        const float x0 = head[i];
        const float x1 = head[i + half];
        head[i] = x0 * c - x1 * s;
        head[i + half] = x0 * s + x1 * c;
    }
}

void softmax(std::span<float> x) {
    const float max = *std::max_element(x.begin(), x.end());
    double sum = 0.0;
    for (float& v : x) {
        v = std::exp(v - max);
        sum += v;
    }
    const float scale = static_cast<float>(1.0 / sum);
    for (float& v : x) v *= scale;
}

float silu(float x) { return x / (1.0f + std::exp(-x)); }

std::uint32_t require_u32(const GgufFile& file, const std::string& key) {
    const auto value = file.get<std::uint32_t>(key);
    if (!value) fail("missing metadata '" + key + "'");
    return *value;
}

float require_f32(const GgufFile& file, const std::string& key) {
    const auto value = file.get<float>(key);
    if (!value) fail("missing metadata '" + key + "'");
    return *value;
}

}  // namespace

Model::Model(const GgufFile& file) {
    const auto architecture = file.get<std::string_view>("general.architecture");
    if (architecture != "qwen3") {
        fail("unsupported architecture '" + std::string(architecture.value_or("(none)")) + "'; only qwen3 is supported");
    }
    const std::string prefix = "qwen3.";

    config_.embedding_dim = require_u32(file, prefix + "embedding_length");
    config_.layer_count = require_u32(file, prefix + "block_count");
    config_.head_count = require_u32(file, prefix + "attention.head_count");
    config_.kv_head_count = require_u32(file, prefix + "attention.head_count_kv");
    config_.feed_forward_dim = require_u32(file, prefix + "feed_forward_length");
    config_.context_length = require_u32(file, prefix + "context_length");
    config_.rope_base = require_f32(file, prefix + "rope.freq_base");
    config_.rms_epsilon = require_f32(file, prefix + "attention.layer_norm_rms_epsilon");

    if (config_.embedding_dim == 0 || config_.layer_count == 0 || config_.head_count == 0 ||
        config_.kv_head_count == 0 || config_.feed_forward_dim == 0 || config_.context_length == 0) {
        fail("a model dimension is zero");
    }
    if (config_.head_count % config_.kv_head_count != 0) fail("head_count is not a multiple of head_count_kv");

    // Qwen3 sets the head size explicitly; it is not always embedding_dim / head_count.
    config_.head_dim = file.get<std::uint32_t>(prefix + "attention.key_length")
                           .value_or(config_.embedding_dim / config_.head_count);
    if (config_.head_dim == 0 || config_.head_dim % 2 != 0) fail("head size must be even and non-zero");
    if (file.get<std::uint32_t>(prefix + "attention.value_length").value_or(config_.head_dim) != config_.head_dim) {
        fail("key and value head sizes differ");
    }

    const TensorInfo* embedding = file.tensor("token_embd.weight");
    if (embedding == nullptr || embedding->dims.size() != 2 || embedding->dims[0] != config_.embedding_dim) {
        fail("missing or misshapen token_embd.weight");
    }
    if (embedding->dims[1] == 0 || embedding->dims[1] > 0x7FFFFFFF) fail("invalid vocabulary size");
    config_.vocab_size = static_cast<std::uint32_t>(embedding->dims[1]);

    const std::size_t embd = config_.embedding_dim;
    const std::size_t query_dim = static_cast<std::size_t>(config_.head_count) * config_.head_dim;
    const std::size_t kv_dim = static_cast<std::size_t>(config_.kv_head_count) * config_.head_dim;
    const std::size_t ff = config_.feed_forward_dim;

    token_embedding_ = load_matrix(file, "token_embd.weight", config_.vocab_size, embd);
    // Without a separate output matrix the token embedding is reused for it.
    output_projection_ = file.tensor("output.weight") != nullptr
                             ? load_matrix(file, "output.weight", config_.vocab_size, embd)
                             : token_embedding_;
    output_norm_ = load(file, "output_norm.weight", embd);

    layers_.resize(config_.layer_count);
    for (std::size_t i = 0; i < layers_.size(); ++i) {
        const std::string block = "blk." + std::to_string(i) + ".";
        Layer& layer = layers_[i];
        layer.attention_norm = load(file, block + "attn_norm.weight", embd);
        layer.query = load_matrix(file, block + "attn_q.weight", query_dim, embd);
        layer.key = load_matrix(file, block + "attn_k.weight", kv_dim, embd);
        layer.value = load_matrix(file, block + "attn_v.weight", kv_dim, embd);
        layer.query_norm = load(file, block + "attn_q_norm.weight", config_.head_dim);
        layer.key_norm = load(file, block + "attn_k_norm.weight", config_.head_dim);
        layer.attention_output = load_matrix(file, block + "attn_output.weight", embd, query_dim);
        layer.feed_forward_norm = load(file, block + "ffn_norm.weight", embd);
        layer.gate = load_matrix(file, block + "ffn_gate.weight", ff, embd);
        layer.up = load_matrix(file, block + "ffn_up.weight", ff, embd);
        layer.down = load_matrix(file, block + "ffn_down.weight", embd, ff);
    }
}

std::span<const float> Model::load(const GgufFile& file, std::string_view name, std::size_t expected_elements) {
    const TensorInfo* info = file.tensor(name);
    if (info == nullptr) fail("missing tensor '" + std::string(name) + "'");
    if (info->n_elements() != expected_elements) fail("tensor '" + std::string(name) + "' has an unexpected size");

    const std::byte* bytes = info->data.data();
    switch (info->type) {
        case TensorType::F32:
            if (reinterpret_cast<std::uintptr_t>(bytes) % alignof(float) == 0) {
                return {reinterpret_cast<const float*>(bytes), expected_elements};
            }
            owned_.emplace_back(expected_elements);
            std::memcpy(owned_.back().data(), bytes, expected_elements * sizeof(float));
            return owned_.back();
        case TensorType::F16:
        case TensorType::BF16: {
            std::vector<float>& out = owned_.emplace_back(expected_elements);
            const bool bfloat = info->type == TensorType::BF16;
            for (std::size_t i = 0; i < expected_elements; ++i) {
                std::uint16_t raw;
                std::memcpy(&raw, bytes + i * sizeof(raw), sizeof(raw));
                out[i] = bfloat ? bfloat_to_float(raw) : half_to_float(raw);
            }
            return out;
        }
        default:
            fail("tensor '" + std::string(name) + "' has type " + std::string(to_string(info->type)) +
                 "; only F32, F16 and BF16 weights are supported so far");
    }
}

Matrix Model::load_matrix(const GgufFile& file, std::string_view name, std::size_t rows, std::size_t cols) {
    // GGUF lists the innermost dimension first, so a rows x cols matrix is stored as {cols, rows}.
    const TensorInfo* info = file.tensor(name);
    if (info == nullptr) fail("missing tensor '" + std::string(name) + "'");
    if (info->dims.size() != 2 || info->dims[0] != cols || info->dims[1] != rows) {
        fail("tensor '" + std::string(name) + "' has an unexpected shape");
    }
    return {load(file, name, rows * cols), rows, cols};
}

Session::Session(const Model& model) : model_(model) {
    const ModelConfig& c = model.config();
    const std::size_t query_dim = static_cast<std::size_t>(c.head_count) * c.head_dim;
    key_cache_.resize(c.layer_count);
    value_cache_.resize(c.layer_count);
    hidden_.resize(c.embedding_dim);
    normed_.resize(c.embedding_dim);
    query_.resize(query_dim);
    attended_.resize(query_dim);
    projected_.resize(c.embedding_dim);
    gate_.resize(c.feed_forward_dim);
    up_.resize(c.feed_forward_dim);
    rope_.resize(c.head_dim);
    logits_.resize(c.vocab_size);
}

std::span<const float> Session::eval(Token token) {
    const ModelConfig& c = model_.config();
    if (token < 0 || static_cast<std::uint32_t>(token) >= c.vocab_size) fail("token id out of range");
    if (position_ >= c.context_length) fail("context length exceeded");

    const std::size_t head_dim = c.head_dim;
    const std::size_t kv_dim = static_cast<std::size_t>(c.kv_head_count) * head_dim;
    const std::size_t heads_per_kv = c.head_count / c.kv_head_count;
    const std::size_t length = position_ + 1;  // positions visible to this token, itself included
    const float attention_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Angles are built by repeated multiplication, as llama.cpp does, so the
    // rounding matches.
    const float theta_scale = std::pow(c.rope_base, -2.0f / static_cast<float>(head_dim));
    float theta = static_cast<float>(position_);
    for (std::size_t i = 0; i < head_dim / 2; ++i) {
        rope_[2 * i] = std::cos(theta);
        rope_[2 * i + 1] = std::sin(theta);
        theta *= theta_scale;
    }

    const auto embedding = model_.token_embedding().row(static_cast<std::size_t>(token));
    std::copy(embedding.begin(), embedding.end(), hidden_.begin());
    scores_.resize(length);

    for (std::size_t l = 0; l < model_.layers().size(); ++l) {
        const Model::Layer& layer = model_.layers()[l];
        std::vector<float>& keys = key_cache_[l];
        std::vector<float>& values = value_cache_[l];
        keys.resize(length * kv_dim);
        values.resize(length * kv_dim);
        const std::span<float> key(keys.data() + position_ * kv_dim, kv_dim);
        const std::span<float> value(values.data() + position_ * kv_dim, kv_dim);

        rms_norm(hidden_, layer.attention_norm, c.rms_epsilon, normed_);
        multiply(layer.query, normed_, query_);
        multiply(layer.key, normed_, key);
        multiply(layer.value, normed_, value);

        for (std::size_t h = 0; h < c.head_count; ++h) {
            const std::span<float> q(query_.data() + h * head_dim, head_dim);
            rms_norm(q, layer.query_norm, c.rms_epsilon, q);
            apply_rope(q, rope_);
        }
        for (std::size_t h = 0; h < c.kv_head_count; ++h) {
            const std::span<float> k = key.subspan(h * head_dim, head_dim);
            rms_norm(k, layer.key_norm, c.rms_epsilon, k);
            apply_rope(k, rope_);
        }

        for (std::size_t h = 0; h < c.head_count; ++h) {
            const std::size_t kv_offset = (h / heads_per_kv) * head_dim;
            const float* q = query_.data() + h * head_dim;
            for (std::size_t t = 0; t < length; ++t) {
                scores_[t] = dot(q, keys.data() + t * kv_dim + kv_offset, head_dim) * attention_scale;
            }
            softmax(scores_);

            float* out = attended_.data() + h * head_dim;
            std::fill_n(out, head_dim, 0.0f);
            for (std::size_t t = 0; t < length; ++t) {
                const float weight = scores_[t];
                const float* v = values.data() + t * kv_dim + kv_offset;
                for (std::size_t i = 0; i < head_dim; ++i) out[i] += weight * v[i];
            }
        }

        multiply(layer.attention_output, attended_, projected_);
        for (std::size_t i = 0; i < hidden_.size(); ++i) hidden_[i] += projected_[i];

        rms_norm(hidden_, layer.feed_forward_norm, c.rms_epsilon, normed_);
        multiply(layer.gate, normed_, gate_);
        multiply(layer.up, normed_, up_);
        for (std::size_t i = 0; i < gate_.size(); ++i) gate_[i] = silu(gate_[i]) * up_[i];
        multiply(layer.down, gate_, projected_);
        for (std::size_t i = 0; i < hidden_.size(); ++i) hidden_[i] += projected_[i];
    }

    rms_norm(hidden_, model_.output_norm(), c.rms_epsilon, normed_);
    multiply(model_.output_projection(), normed_, logits_);

    ++position_;
    return logits_;
}

}  // namespace brisk
