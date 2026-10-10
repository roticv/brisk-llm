#include "model.h"

#include "kernels.h"
#include "kernels_f32.h"

#include <algorithm>
#include <chrono>
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

void embedding_row(const Matrix& m, std::size_t r, std::span<float> out) {
    const bool grouped = m.format == WeightFormat::Q4_0x4 || m.format == WeightFormat::TQ2_0x4 ||
                         m.format == WeightFormat::Q4_Kx4 || m.format == WeightFormat::Q6_Kx4;
    const std::byte* row = m.format == WeightFormat::F32 || grouped ? nullptr : m.row(r);
    switch (m.format) {
        case WeightFormat::F32: {
            const auto f32 = m.f32_row(r);
            std::copy(f32.begin(), f32.end(), out.begin());
            break;
        }
        case WeightFormat::Q4_0: dequantize_q4_0(reinterpret_cast<const BlockQ4_0*>(row), out.data(), m.cols); break;
        case WeightFormat::Q4_0x4:
            dequantize_q4_0x4_row(reinterpret_cast<const BlockQ4_0x4*>(m.bytes.data() + (r / 4) * 4 * m.row_bytes()),
                                  r % 4, out.data(), m.cols);
            break;
        case WeightFormat::Q4_1: dequantize_q4_1(reinterpret_cast<const BlockQ4_1*>(row), out.data(), m.cols); break;
        case WeightFormat::Q4_K: dequantize_q4_k(reinterpret_cast<const BlockQ4_K*>(row), out.data(), m.cols); break;
        case WeightFormat::Q6_K: dequantize_q6_k(reinterpret_cast<const BlockQ6_K*>(row), out.data(), m.cols); break;
        case WeightFormat::TQ2_0: dequantize_tq2_0(reinterpret_cast<const BlockTQ2_0*>(row), out.data(), m.cols); break;
        case WeightFormat::TQ2_0x4:
            dequantize_tq2_0x4_row(reinterpret_cast<const BlockTQ2_0x4*>(m.bytes.data() + (r / 4) * 4 * m.row_bytes()),
                                   r % 4, out.data(), m.cols);
            break;
        case WeightFormat::Q4_Kx4:
            dequantize_q4_kx4_row(reinterpret_cast<const BlockQ4_Kx4*>(m.bytes.data() + (r / 4) * 4 * m.row_bytes()),
                                  r % 4, out.data(), m.cols);
            break;
        case WeightFormat::Q6_Kx4:
            dequantize_q6_kx4_row(reinterpret_cast<const BlockQ6_Kx4*>(m.bytes.data() + (r / 4) * 4 * m.row_bytes()),
                                  r % 4, out.data(), m.cols);
            break;
    }
}

std::optional<WeightFormat> weight_format(TensorType type) {
    switch (type) {
        case TensorType::Q4_0: return WeightFormat::Q4_0;
        case TensorType::Q4_1: return WeightFormat::Q4_1;
        case TensorType::Q4_K: return WeightFormat::Q4_K;
        case TensorType::Q6_K: return WeightFormat::Q6_K;
        case TensorType::TQ2_0: return WeightFormat::TQ2_0;
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
    if (architecture == "qwen3") config_.architecture = Architecture::Qwen3;
    else if (architecture == "bitnet" || architecture == "bitnet-b1.58") config_.architecture = Architecture::BitNet;
    else fail("unsupported architecture '" + std::string(architecture.value_or("(none)")) + "'; qwen3 and bitnet are supported");
    const std::string prefix = std::string(*architecture) + ".";
    const bool bitnet = config_.architecture == Architecture::BitNet;

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
    if (config_.head_count / config_.kv_head_count > 8) fail("more than 8 query heads per key/value head is not supported");

    // Qwen3 sets the head size explicitly; it is not always embedding_dim / head_count.
    config_.head_dim = file.get<std::uint32_t>(prefix + "attention.key_length")
                           .value_or(config_.embedding_dim / config_.head_count);
    if (config_.head_dim == 0 || config_.head_dim % 32 != 0) fail("head size must be a non-zero multiple of 32");
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
        if (bitnet) {
            layer.attention_sub_norm = load(file, block + "attn_sub_norm.weight", embd);
            layer.feed_forward_sub_norm = load(file, block + "ffn_sub_norm.weight", ff);
        } else {
            layer.query_norm = load(file, block + "attn_q_norm.weight", config_.head_dim);
            layer.key_norm = load(file, block + "attn_k_norm.weight", config_.head_dim);
        }
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
        if (m.format == WeightFormat::Q4_0 && rows % 4 == 0 && kernels::prefers_q4_0x4()) {
            // Interleave four rows per block for the NEON matrix kernels.
            const std::size_t blocks = cols / kBlockSize;
            std::vector<std::byte>& packed = packed_.emplace_back(rows / 4 * blocks * sizeof(BlockQ4_0x4));
            repack_q4_0x4(reinterpret_cast<const BlockQ4_0*>(info->data.data()), rows, blocks,
                          reinterpret_cast<BlockQ4_0x4*>(packed.data()));
            m.format = WeightFormat::Q4_0x4;
            m.bytes = packed;
        } else if (m.format == WeightFormat::TQ2_0 && rows % 4 == 0 && kernels::prefers_q4_0x4()) {
            const std::size_t blocks = cols / kSuperBlockSize;
            std::vector<std::byte>& packed = packed_.emplace_back(rows / 4 * blocks * sizeof(BlockTQ2_0x4));
            repack_tq2_0x4(reinterpret_cast<const BlockTQ2_0*>(info->data.data()), rows, blocks,
                           reinterpret_cast<BlockTQ2_0x4*>(packed.data()));
            m.format = WeightFormat::TQ2_0x4;
            m.bytes = packed;
        } else if (m.format == WeightFormat::Q4_K && rows % 4 == 0 && kernels::prefers_q4_0x4()) {
            const std::size_t blocks = cols / kSuperBlockSize;
            std::vector<std::byte>& packed = packed_.emplace_back(rows / 4 * blocks * sizeof(BlockQ4_Kx4));
            repack_q4_kx4(reinterpret_cast<const BlockQ4_K*>(info->data.data()), rows, blocks,
                          reinterpret_cast<BlockQ4_Kx4*>(packed.data()));
            m.format = WeightFormat::Q4_Kx4;
            m.bytes = packed;
        } else if (m.format == WeightFormat::Q6_K && rows % 4 == 0 && kernels::prefers_q4_0x4()) {
            const std::size_t blocks = cols / kSuperBlockSize;
            std::vector<std::byte>& packed = packed_.emplace_back(rows / 4 * blocks * sizeof(BlockQ6_Kx4));
            repack_q6_kx4(reinterpret_cast<const BlockQ6_K*>(info->data.data()), rows, blocks,
                          reinterpret_cast<BlockQ6_Kx4*>(packed.data()));
            m.format = WeightFormat::Q6_Kx4;
            m.bytes = packed;
        }
    } else {
        m.f32 = load(file, name, rows * cols);
    }
    return m;
}

// The most tokens processed in one pass; bounds the scratch memory.
constexpr std::size_t kMaxBatch = 256;

// Adds the time from its construction to its destruction to a profile counter.
class Timed {
public:
    explicit Timed(double& counter) : counter_(counter), start_(std::chrono::steady_clock::now()) {}
    ~Timed() { counter_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count(); }

private:
    double& counter_;
    std::chrono::steady_clock::time_point start_;
};

void Session::multiply(const Matrix& m, const float* x, std::size_t n, float* out) {
    const Target target{&m, out};
    multiply(std::span<const Target>(&target, 1), x, n);
}

void Session::multiply(std::span<const Target> targets, const float* x, std::size_t n) {
    const Timed timed(profile_.matmul);
    const std::size_t cols = targets[0].matrix->cols;

    // Row chunks per target: enough tasks that the threads stay balanced, but
    // each large enough that handing it out costs nothing next to doing it.
    // Chunks are multiples of 4 rows so the grouped layouts never split a group.
    // Single-row passes stream the weights, so big chunks keep the streams long;
    // batched passes are compute-bound, so smaller chunks balance the threads.
    struct Plan {
        std::size_t chunk;
        std::size_t first_task;
        const void* activations;
        kernels::KernelSet kernels;
    };
    std::vector<Plan> plans(targets.size());
    std::size_t total_tasks = 0;
    for (std::size_t i = 0; i < targets.size(); ++i) {
        const Matrix& m = *targets[i].matrix;
        if (m.cols != cols) fail("fused multiply needs matrices with the same input size");
        plans[i].chunk = (n == 1 ? std::clamp<std::size_t>(m.rows / (pool_.size() * 8), 16, 256)
                                 : std::clamp<std::size_t>(m.rows / (pool_.size() * 4), 4, 64)) / 4 * 4;
        plans[i].first_task = total_tasks;
        total_tasks += (m.rows + plans[i].chunk - 1) / plans[i].chunk;
    }

    // Quantise the activations once per format the targets take.
    const auto quantize_start = std::chrono::steady_clock::now();
    bool have_q8_0 = false, have_q8_k = false;
    for (const Target& t : targets) {
        if (t.matrix->format == WeightFormat::F32) continue;
        const std::size_t blocks = cols / block_elements(t.matrix->format);
        if (takes_q8_k(t.matrix->format) && !have_q8_k) {
            q8_k_.resize(n * blocks);
            pool_.run(n, [&](std::size_t r) { quantize_q8_k(x + r * cols, q8_k_.data() + r * blocks, cols); });
            have_q8_k = true;
        } else if (!takes_q8_k(t.matrix->format) && !have_q8_0) {
            q8_0_.resize(n * blocks);
            pool_.run(n, [&](std::size_t r) { quantize_q8_0(x + r * cols, q8_0_.data() + r * blocks, cols); });
            have_q8_0 = true;
        }
    }
    profile_.quantize += std::chrono::duration<double>(std::chrono::steady_clock::now() - quantize_start).count();

    // Kernels and, for batched passes, their rearranged activations (shared
    // between targets with the same kernel set; at most two distinct sets).
    std::size_t prepared_used = 0;
    for (std::size_t i = 0; i < targets.size(); ++i) {
        const Matrix& m = *targets[i].matrix;
        if (m.format == WeightFormat::F32) continue;
        plans[i].kernels = kernels::kernels_for(m.format);
        plans[i].activations = takes_q8_k(m.format) ? static_cast<const void*>(q8_k_.data())
                                                    : static_cast<const void*>(q8_0_.data());
        if (n > 1 && plans[i].kernels.prepare != nullptr) {
            bool shared = false;
            for (std::size_t j = 0; j < i && !shared; ++j) {
                if (plans[j].kernels.prepare == plans[i].kernels.prepare && targets[j].matrix->format == m.format) {
                    plans[i].activations = plans[j].activations;
                    shared = true;
                }
            }
            if (!shared) {
                if (prepared_used == 2) fail("fused multiply supports at most two kernel sets");
                const std::size_t blocks = cols / block_elements(m.format);
                std::vector<std::byte>& buffer = prepared_[prepared_used++];
                buffer.resize(plans[i].kernels.prepared_bytes(n, blocks));
                plans[i].kernels.prepare(plans[i].activations, n, blocks, buffer.data());
                plans[i].activations = buffer.data();
            }
        }
    }

    pool_.run(total_tasks, [&](std::size_t task) {
        std::size_t i = targets.size() - 1;
        while (plans[i].first_task > task) --i;
        const Matrix& m = *targets[i].matrix;
        float* out = targets[i].out;
        const Plan& plan = plans[i];
        const std::size_t begin = (task - plan.first_task) * plan.chunk;
        const std::size_t end = std::min(m.rows, begin + plan.chunk);

        if (m.format == WeightFormat::F32) {
            for (std::size_t r = begin; r < end; ++r) {
                for (std::size_t t = 0; t < n; ++t) {
                    out[t * m.rows + r] = kernels::dot_f32(m.f32.data() + r * m.cols, x + t * m.cols, m.cols);
                }
            }
        } else if (n == 1) {
            plan.kernels.matvec(m.row(begin), end - begin, m.cols, plan.activations, out + begin);
        } else {
            plan.kernels.matmul(m.row(begin), end - begin, m.cols, plan.activations, n, out + begin, m.rows);
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

std::span<const float> Session::eval_all(std::span<const Token> tokens) {
    const ModelConfig& c = model_.config();
    if (tokens.empty() || tokens.size() > kMaxScoredTokens) fail("eval_all takes 1 to 64 tokens");
    for (const Token token : tokens) {
        if (token < 0 || static_cast<std::uint32_t>(token) >= c.vocab_size) fail("token id out of range");
    }
    if (tokens.size() > c.context_length - position_) fail("context length exceeded");
    eval_batch(tokens, true);
    return std::span<const float>(logits_.data(), tokens.size() * c.vocab_size);
}

void Session::rollback(std::size_t position) {
    if (position > position_) fail("cannot roll forward");
    position_ = position;  // the caches keep their storage; later entries are simply overwritten
}

double Session::log_likelihood(std::span<const Token> tokens) {
    const ModelConfig& c = model_.config();
    if (tokens.size() < 2) fail("need at least two tokens to score");
    for (const Token token : tokens) {
        if (token < 0 || static_cast<std::uint32_t>(token) >= c.vocab_size) fail("token id out of range");
    }
    if (tokens.size() > c.context_length - position_) fail("context length exceeded");

    // Smaller batches than eval(): every token's logits are kept, vocab_size floats each.
    constexpr std::size_t kScoreBatch = kMaxScoredTokens;
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

        {
            const Timed timed(profile_.norm_rope);
            pool_.run(n, [&](std::size_t t) {
                rms_norm(rows(hidden_, t, embd), layer.attention_norm, c.rms_epsilon, rows(normed_, t, embd));
            });
        }
        {
            const Target qkv[3] = {{&layer.query, query_.data()}, {&layer.key, new_keys}, {&layer.value, new_values}};
            multiply(qkv, normed_.data(), n);
        }

        Timed* phase = new Timed(profile_.norm_rope);
        pool_.run(n, [&](std::size_t t) {
            const std::span<const float> rope(rope_.data() + t * head_dim, head_dim);
            for (std::size_t h = 0; h < c.head_count; ++h) {
                const std::span<float> q(query_.data() + t * query_dim + h * head_dim, head_dim);
                if (!layer.query_norm.empty()) rms_norm(q, layer.query_norm, c.rms_epsilon, q);
                apply_rope(q, rope);
            }
            for (std::size_t h = 0; h < c.kv_head_count; ++h) {
                const std::span<float> k(new_keys + t * kv_dim + h * head_dim, head_dim);
                if (!layer.key_norm.empty()) rms_norm(k, layer.key_norm, c.rms_epsilon, k);
                apply_rope(k, rope);
            }
        });

        delete phase;
        phase = new Timed(profile_.attention);
        // Causal attention: token t of the batch sees positions 0 to position_ + t.
        // One task covers a key/value head and up to kTokenTile consecutive
        // tokens, so each cached row is read once for all their query heads.
        constexpr std::size_t kTokenTile = 4;
        const std::size_t token_tiles = (n + kTokenTile - 1) / kTokenTile;
        pool_.run(token_tiles * c.kv_head_count, [&](std::size_t task) {
            const std::size_t tile = task / c.kv_head_count;
            const std::size_t kv = task % c.kv_head_count;
            const std::size_t t0 = tile * kTokenTile;
            const std::size_t t1 = std::min(n, t0 + kTokenTile);
            const std::size_t kv_offset = kv * head_dim;
            const std::size_t visible_last = position_ + t1;  // positions seen by the tile's last token
            // Scores: one row per (token, query head) of the tile.
            float* scores = scores_.data() + (t0 * c.head_count + kv * heads_per_kv) * length;
            auto score_row = [&](std::size_t t, std::size_t g) {
                return scores + ((t - t0) * c.head_count + g) * length;
            };

            // All the tile's queries (each token's heads for this key/value
            // head) against every position the last token sees; positions a
            // token cannot see are computed too but ignored by its softmax.
            const float* queries[kTokenTile * 8];
            float* rows_out[kTokenTile * 8];
            std::size_t nq = 0;
            for (std::size_t t = t0; t < t1; ++t) {
                for (std::size_t g = 0; g < heads_per_kv; ++g) {
                    queries[nq] = query_.data() + t * query_dim + (kv * heads_per_kv + g) * head_dim;
                    rows_out[nq] = score_row(t, g);
                    ++nq;
                }
            }
            kernels::scores_f32(queries, nq, keys.data() + kv_offset, kv_dim, visible_last, head_dim, attention_scale,
                                rows_out);
            for (std::size_t t = t0; t < t1; ++t) {
                const std::size_t visible = position_ + t + 1;
                for (std::size_t g = 0; g < heads_per_kv; ++g) {
                    kernels::softmax_f32(std::span<float>(score_row(t, g), visible));
                }
                kernels::weighted_sum_f32(values.data() + kv_offset, kv_dim, visible, score_row(t, 0), length,
                                          heads_per_kv, head_dim,
                                          attended_.data() + t * query_dim + kv * heads_per_kv * head_dim);
            }
        });
        delete phase;
        if (!layer.attention_sub_norm.empty()) {
            const Timed timed(profile_.norm_rope);
            pool_.run(n, [&](std::size_t t) {
                const std::span<float> a(attended_.data() + t * query_dim, query_dim);
                rms_norm(a, layer.attention_sub_norm, c.rms_epsilon, a);
            });
        }
        multiply(layer.attention_output, attended_.data(), n, projected_.data());
        {
            const Timed timed(profile_.norm_rope);
            pool_.run(n, [&](std::size_t t) {
                float* h = hidden_.data() + t * embd;
                const float* p = projected_.data() + t * embd;
                for (std::size_t i = 0; i < embd; ++i) h[i] += p[i];
                rms_norm(rows(hidden_, t, embd), layer.feed_forward_norm, c.rms_epsilon, rows(normed_, t, embd));
            });
        }
        {
            const Target gate_up[2] = {{&layer.gate, gate_.data()}, {&layer.up, up_.data()}};
            multiply(gate_up, normed_.data(), n);
        }
        {
            const Timed timed(profile_.activation);
            pool_.run(n, [&](std::size_t t) {
                float* g = gate_.data() + t * c.feed_forward_dim;
                const float* u = up_.data() + t * c.feed_forward_dim;
                if (c.architecture == Architecture::BitNet) {
                    for (std::size_t i = 0; i < c.feed_forward_dim; ++i) {
                        const float r = std::max(g[i], 0.0f);
                        g[i] = r * r * u[i];
                    }
                } else {
                    for (std::size_t i = 0; i < c.feed_forward_dim; ++i) g[i] = silu(g[i]) * u[i];
                }
                if (!layer.feed_forward_sub_norm.empty()) {
                    const std::span<float> gs(g, c.feed_forward_dim);
                    rms_norm(gs, layer.feed_forward_sub_norm, c.rms_epsilon, gs);
                }
            });
        }
        multiply(layer.down, gate_.data(), n, projected_.data());
        {
            const Timed timed(profile_.norm_rope);
            pool_.run(n, [&](std::size_t t) {
                float* h = hidden_.data() + t * embd;
                const float* p = projected_.data() + t * embd;
                for (std::size_t i = 0; i < embd; ++i) h[i] += p[i];
            });
        }
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
