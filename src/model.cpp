#include "model.h"

#include "kernels.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>

namespace brisk {

namespace {

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("model: " + what); }

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

void embedding_row(const Matrix& m, std::size_t r, std::span<float> out) {
    const std::byte* row = m.format == WeightFormat::F32 ? nullptr : m.row(r);
    switch (m.format) {
        case WeightFormat::F32: {
            const auto f32 = m.f32_row(r);
            std::copy(f32.begin(), f32.end(), out.begin());
            break;
        }
        case WeightFormat::Q4_0: dequantize_q4_0(reinterpret_cast<const BlockQ4_0*>(row), out.data(), m.cols); break;
        case WeightFormat::Q4_1: dequantize_q4_1(reinterpret_cast<const BlockQ4_1*>(row), out.data(), m.cols); break;
        case WeightFormat::Q4_K: dequantize_q4_k(reinterpret_cast<const BlockQ4_K*>(row), out.data(), m.cols); break;
        case WeightFormat::Q6_K: dequantize_q6_k(reinterpret_cast<const BlockQ6_K*>(row), out.data(), m.cols); break;
    }
}

std::optional<WeightFormat> weight_format(TensorType type) {
    switch (type) {
        case TensorType::Q4_0: return WeightFormat::Q4_0;
        case TensorType::Q4_1: return WeightFormat::Q4_1;
        case TensorType::Q4_K: return WeightFormat::Q4_K;
        case TensorType::Q6_K: return WeightFormat::Q6_K;
        default: return std::nullopt;
    }
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
                 "; only F32, F16 and BF16 are supported for this tensor");
    }
}

Matrix Model::load_matrix(const GgufFile& file, std::string_view name, std::size_t rows, std::size_t cols) {
    // GGUF lists the innermost dimension first, so a rows x cols matrix is stored as {cols, rows}.
    const TensorInfo* info = file.tensor(name);
    if (info == nullptr) fail("missing tensor '" + std::string(name) + "'");
    if (info->dims.size() != 2 || info->dims[0] != cols || info->dims[1] != rows) {
        fail("tensor '" + std::string(name) + "' has an unexpected shape");
    }

    Matrix m;
    m.rows = rows;
    m.cols = cols;
    if (const auto format = weight_format(info->type)) {
        m.format = *format;
        if (cols % block_elements(*format) != 0) {
            fail("tensor '" + std::string(name) + "' row length is not a multiple of its block size");
        }
        if (info->data.size() != rows * m.row_bytes() || reinterpret_cast<std::uintptr_t>(info->data.data()) % 4 != 0) {
            fail("tensor '" + std::string(name) + "' has unexpected quantised data");
        }
        m.bytes = info->data;
    } else {
        m.f32 = load(file, name, rows * cols);
    }
    return m;
}

// The most tokens processed in one pass; bounds the scratch memory.
constexpr std::size_t kMaxBatch = 256;

void Session::multiply(const Matrix& m, const float* x, std::size_t n, float* out) {
    // Enough tasks that the threads stay balanced, but each large enough that
    // handing it out costs nothing next to doing it.
    const std::size_t chunk = std::clamp<std::size_t>(m.rows / (pool_.size() * 8), 16, 256);
    const std::size_t tasks = (m.rows + chunk - 1) / chunk;

    if (m.format == WeightFormat::F32) {
        pool_.run(tasks, [&](std::size_t task) {
            const std::size_t end = std::min(m.rows, (task + 1) * chunk);
            for (std::size_t r = task * chunk; r < end; ++r) {
                for (std::size_t t = 0; t < n; ++t) {
                    out[t * m.rows + r] = dot(m.f32.data() + r * m.cols, x + t * m.cols, m.cols);
                }
            }
        });
        return;
    }

    // Quantise the activations to the format the weights' kernels take.
    const std::size_t blocks = m.cols / block_elements(m.format);
    const void* activations = nullptr;
    if (takes_q8_k(m.format)) {
        q8_k_.resize(n * blocks);
        pool_.run(n, [&](std::size_t t) { quantize_q8_k(x + t * m.cols, q8_k_.data() + t * blocks, m.cols); });
        activations = q8_k_.data();
    } else {
        q8_0_.resize(n * blocks);
        pool_.run(n, [&](std::size_t t) { quantize_q8_0(x + t * m.cols, q8_0_.data() + t * blocks, m.cols); });
        activations = q8_0_.data();
    }

    const kernels::KernelSet k = kernels::kernels_for(m.format);
    if (n == 1) {
        pool_.run(tasks, [&](std::size_t task) {
            const std::size_t begin = task * chunk;
            const std::size_t end = std::min(m.rows, begin + chunk);
            k.matvec(m.row(begin), end - begin, m.cols, activations, out + begin);
        });
        return;
    }
    pool_.run(tasks, [&](std::size_t task) {
        const std::size_t begin = task * chunk;
        const std::size_t end = std::min(m.rows, begin + chunk);
        // The kernel writes a dense (n x rows-in-chunk) block; scatter it into out's stride.
        float local[kMaxBatch * 256];
        k.matmul(m.row(begin), end - begin, m.cols, activations, n, local);
        for (std::size_t t = 0; t < n; ++t) {
            std::copy_n(local + t * (end - begin), end - begin, out + t * m.rows + begin);
        }
    });
}

Session::Session(const Model& model, std::size_t threads) : model_(model), pool_(std::max<std::size_t>(threads, 1)) {
    const ModelConfig& c = model.config();
    const std::size_t query_dim = static_cast<std::size_t>(c.head_count) * c.head_dim;
    key_cache_.resize(c.layer_count);
    value_cache_.resize(c.layer_count);
    hidden_.resize(kMaxBatch * c.embedding_dim);
    normed_.resize(kMaxBatch * c.embedding_dim);
    query_.resize(kMaxBatch * query_dim);
    attended_.resize(kMaxBatch * query_dim);
    projected_.resize(kMaxBatch * c.embedding_dim);
    gate_.resize(kMaxBatch * c.feed_forward_dim);
    up_.resize(kMaxBatch * c.feed_forward_dim);
    rope_.resize(kMaxBatch * c.head_dim);
    logits_.resize(c.vocab_size);
}

std::span<const float> Session::eval(std::span<const Token> tokens) {
    const ModelConfig& c = model_.config();
    if (tokens.empty()) fail("no tokens to evaluate");
    for (const Token token : tokens) {
        if (token < 0 || static_cast<std::uint32_t>(token) >= c.vocab_size) fail("token id out of range");
    }
    if (tokens.size() > c.context_length - position_) fail("context length exceeded");

    for (std::size_t begin = 0; begin < tokens.size(); begin += kMaxBatch) {
        eval_batch(tokens.subspan(begin, std::min(kMaxBatch, tokens.size() - begin)), false);
    }
    return std::span<const float>(logits_.data(), c.vocab_size);
}

double Session::log_likelihood(std::span<const Token> tokens) {
    const ModelConfig& c = model_.config();
    if (tokens.size() < 2) fail("need at least two tokens to score");
    for (const Token token : tokens) {
        if (token < 0 || static_cast<std::uint32_t>(token) >= c.vocab_size) fail("token id out of range");
    }
    if (tokens.size() > c.context_length - position_) fail("context length exceeded");

    // Smaller batches than eval(): every token's logits are kept, vocab_size floats each.
    constexpr std::size_t kScoreBatch = 64;
    double total = 0.0;
    for (std::size_t begin = 0; begin < tokens.size(); begin += kScoreBatch) {
        const std::size_t n = std::min(kScoreBatch, tokens.size() - begin);
        eval_batch(tokens.subspan(begin, n), true);
        // Row t predicts tokens[begin + t + 1]; the last row of the text predicts nothing.
        for (std::size_t t = 0; t < n && begin + t + 1 < tokens.size(); ++t) {
            const float* row = logits_.data() + t * c.vocab_size;
            const float max = *std::max_element(row, row + c.vocab_size);
            double sum = 0.0;
            for (std::size_t i = 0; i < c.vocab_size; ++i) sum += std::exp(static_cast<double>(row[i] - max));
            total += static_cast<double>(row[tokens[begin + t + 1]] - max) - std::log(sum);
        }
    }
    return total;
}

void Session::eval_batch(std::span<const Token> tokens, bool all_logits) {
    const ModelConfig& c = model_.config();
    const std::size_t n = tokens.size();
    const std::size_t embd = c.embedding_dim;
    const std::size_t head_dim = c.head_dim;
    const std::size_t query_dim = static_cast<std::size_t>(c.head_count) * head_dim;
    const std::size_t kv_dim = static_cast<std::size_t>(c.kv_head_count) * head_dim;
    const std::size_t heads_per_kv = c.head_count / c.kv_head_count;
    const std::size_t length = position_ + n;  // positions visible to the last token of the batch
    const float attention_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Angles are built by repeated multiplication, as llama.cpp does, so the
    // rounding matches.
    const float theta_scale = std::pow(c.rope_base, -2.0f / static_cast<float>(head_dim));
    for (std::size_t t = 0; t < n; ++t) {
        float theta = static_cast<float>(position_ + t);
        float* rope = rope_.data() + t * head_dim;
        for (std::size_t i = 0; i < head_dim / 2; ++i) {
            rope[2 * i] = std::cos(theta);
            rope[2 * i + 1] = std::sin(theta);
            theta *= theta_scale;
        }
    }

    for (std::size_t t = 0; t < n; ++t) {
        embedding_row(model_.token_embedding(), static_cast<std::size_t>(tokens[t]),
                      std::span<float>(hidden_.data() + t * embd, embd));
    }
    scores_.resize(n * c.head_count * length);

    auto rows = [](std::vector<float>& v, std::size_t t, std::size_t width) {
        return std::span<float>(v.data() + t * width, width);
    };

    for (std::size_t l = 0; l < model_.layers().size(); ++l) {
        const Model::Layer& layer = model_.layers()[l];
        std::vector<float>& keys = key_cache_[l];
        std::vector<float>& values = value_cache_[l];
        keys.resize(length * kv_dim);
        values.resize(length * kv_dim);
        float* new_keys = keys.data() + position_ * kv_dim;  // this batch's rows, contiguous
        float* new_values = values.data() + position_ * kv_dim;

        pool_.run(n, [&](std::size_t t) {
            rms_norm(rows(hidden_, t, embd), layer.attention_norm, c.rms_epsilon, rows(normed_, t, embd));
        });
        multiply(layer.query, normed_.data(), n, query_.data());
        multiply(layer.key, normed_.data(), n, new_keys);
        multiply(layer.value, normed_.data(), n, new_values);

        pool_.run(n, [&](std::size_t t) {
            const std::span<const float> rope(rope_.data() + t * head_dim, head_dim);
            for (std::size_t h = 0; h < c.head_count; ++h) {
                const std::span<float> q(query_.data() + t * query_dim + h * head_dim, head_dim);
                rms_norm(q, layer.query_norm, c.rms_epsilon, q);
                apply_rope(q, rope);
            }
            for (std::size_t h = 0; h < c.kv_head_count; ++h) {
                const std::span<float> k(new_keys + t * kv_dim + h * head_dim, head_dim);
                rms_norm(k, layer.key_norm, c.rms_epsilon, k);
                apply_rope(k, rope);
            }
        });

        // Causal attention: token t of the batch sees positions 0 to position_ + t.
        // One task covers the query heads that share a key/value head, so each
        // cached row is read once for all of them.
        pool_.run(n * c.kv_head_count, [&](std::size_t task) {
            const std::size_t t = task / c.kv_head_count;
            const std::size_t kv = task % c.kv_head_count;
            const std::size_t visible = position_ + t + 1;
            const std::size_t kv_offset = kv * head_dim;
            const float* q = query_.data() + t * query_dim + kv * heads_per_kv * head_dim;
            float* scores = scores_.data() + task * heads_per_kv * length;  // one row per query head

            for (std::size_t p = 0; p < visible; ++p) {
                const float* k = keys.data() + p * kv_dim + kv_offset;
                for (std::size_t g = 0; g < heads_per_kv; ++g) {
                    scores[g * length + p] = dot(q + g * head_dim, k, head_dim) * attention_scale;
                }
            }
            for (std::size_t g = 0; g < heads_per_kv; ++g) softmax(std::span<float>(scores + g * length, visible));

            float* out = attended_.data() + t * query_dim + kv * heads_per_kv * head_dim;
            std::fill_n(out, heads_per_kv * head_dim, 0.0f);
            for (std::size_t p = 0; p < visible; ++p) {
                const float* v = values.data() + p * kv_dim + kv_offset;
                for (std::size_t g = 0; g < heads_per_kv; ++g) {
                    const float weight = scores[g * length + p];
                    float* o = out + g * head_dim;
                    for (std::size_t i = 0; i < head_dim; ++i) o[i] += weight * v[i];
                }
            }
        });

        multiply(layer.attention_output, attended_.data(), n, projected_.data());
        pool_.run(n, [&](std::size_t t) {
            float* h = hidden_.data() + t * embd;
            const float* p = projected_.data() + t * embd;
            for (std::size_t i = 0; i < embd; ++i) h[i] += p[i];
            rms_norm(rows(hidden_, t, embd), layer.feed_forward_norm, c.rms_epsilon, rows(normed_, t, embd));
        });
        multiply(layer.gate, normed_.data(), n, gate_.data());
        multiply(layer.up, normed_.data(), n, up_.data());
        pool_.run(n, [&](std::size_t t) {
            float* g = gate_.data() + t * c.feed_forward_dim;
            const float* u = up_.data() + t * c.feed_forward_dim;
            for (std::size_t i = 0; i < c.feed_forward_dim; ++i) g[i] = silu(g[i]) * u[i];
        });
        multiply(layer.down, gate_.data(), n, projected_.data());
        pool_.run(n, [&](std::size_t t) {
            float* h = hidden_.data() + t * embd;
            const float* p = projected_.data() + t * embd;
            for (std::size_t i = 0; i < embd; ++i) h[i] += p[i];
        });
    }

    if (all_logits) {
        pool_.run(n, [&](std::size_t t) {
            rms_norm(rows(hidden_, t, embd), model_.output_norm(), c.rms_epsilon, rows(normed_, t, embd));
        });
        logits_.resize(n * c.vocab_size);
        multiply(model_.output_projection(), normed_.data(), n, logits_.data());
    } else {
        // Only the last token's logits are needed.
        rms_norm(rows(hidden_, n - 1, embd), model_.output_norm(), c.rms_epsilon, rows(normed_, 0, embd));
        multiply(model_.output_projection(), normed_.data(), 1, logits_.data());
    }

    position_ += n;
}

}  // namespace brisk
