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
    const bool grouped = m.format == WeightFormat::Q4_0x4 || m.format == WeightFormat::Q5_0x4 || m.format == WeightFormat::TQ2_0x4 ||
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
        case WeightFormat::Q5_0: dequantize_q5_0(reinterpret_cast<const BlockQ5_0*>(row), out.data(), m.cols); break;
        case WeightFormat::Q5_0x4:
            dequantize_q5_0x4_row(reinterpret_cast<const BlockQ5_0x4*>(m.bytes.data() + (r / 4) * 4 * m.row_bytes()),
                                  r % 4, out.data(), m.cols);
            break;
        case WeightFormat::Q8_0: dequantize_q8_0(reinterpret_cast<const BlockQ8_0*>(row), out.data(), m.cols); break;
        case WeightFormat::Q4_K: dequantize_q4_k(reinterpret_cast<const BlockQ4_K*>(row), out.data(), m.cols); break;
        case WeightFormat::Q5_K: dequantize_q5_k(reinterpret_cast<const BlockQ5_K*>(row), out.data(), m.cols); break;
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
        case TensorType::Q5_0: return WeightFormat::Q5_0;
        case TensorType::Q8_0: return WeightFormat::Q8_0;
        case TensorType::Q4_K: return WeightFormat::Q4_K;
        case TensorType::Q5_K: return WeightFormat::Q5_K;
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
    if (weight.empty()) {  // no weight: plain normalisation (Gemma 4's V)
        for (std::size_t i = 0; i < x.size(); ++i) out[i] = x[i] * scale;
        return;
    }
    for (std::size_t i = 0; i < x.size(); ++i) out[i] = x[i] * scale * weight[i];
}

// Rotary position embedding. `rope` holds cos, sin for each pair; in the
// "NeoX" layout element i is paired with element i + dim/2, in the "normal"
// one element 2i with 2i + 1.
// Only the first `rope.size()` dimensions rotate (all of them unless the
// model rotates part of the head, as Phi does).
void apply_rope(std::span<float> head, std::span<const float> rope, bool neox) {
    const std::size_t half = rope.size() / 2;
    for (std::size_t i = 0; i < half; ++i) {
        const float c = rope[2 * i];
        const float s = rope[2 * i + 1];
        float& a = head[neox ? i : 2 * i];
        float& b = head[neox ? i + half : 2 * i + 1];
        const float x0 = a, x1 = b;
        a = x0 * c - x1 * s;
        b = x0 * s + x1 * c;
    }
}

void add_bias(std::span<float> x, std::span<const float> bias) {
    for (std::size_t i = 0; i < x.size(); ++i) x[i] += bias[i];
}



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

Architecture architecture_of(const GgufFile& file) {
    const auto architecture = file.get<std::string_view>("general.architecture");
    if (architecture == "qwen3") return Architecture::Qwen3;
    if (architecture == "qwen2") return Architecture::Qwen2;
    if (architecture == "llama") return Architecture::Llama;
    if (architecture == "gemma3") return Architecture::Gemma3;
    if (architecture == "gemma4") return Architecture::Gemma4;
    if (architecture == "bitnet" || architecture == "bitnet-b1.58") return Architecture::BitNet;
    if (architecture == "phi3") return Architecture::Phi3;
    if (architecture == "lfm2") return Architecture::LFM2;
    fail("unsupported architecture '" + std::string(architecture.value_or("(none)")) +
         "'; qwen3, qwen2, llama, gemma3, gemma4, bitnet, phi3 and lfm2 are supported");
}

// A float32 tensor of the file, in place (the config's rope factors).
std::span<const float> f32_tensor(const GgufFile& file, std::string_view name, std::size_t count) {
    const TensorInfo* info = file.tensor(name);
    if (info == nullptr) fail("missing tensor '" + std::string(name) + "'");
    if (info->type != TensorType::F32 || info->n_elements() != count) fail("tensor '" + std::string(name) + "' is not float32 of the expected size");
    return {reinterpret_cast<const float*>(info->data.data()), count};
}

ModelConfig Model::parse_config(const GgufFile& file) {
    ModelConfig config;
    config.architecture = architecture_of(file);
    const auto architecture = file.get<std::string_view>("general.architecture");
    const std::string prefix = std::string(*architecture) + ".";
    const bool bitnet = config.architecture == Architecture::BitNet;
    config.rope_neox = config.architecture != Architecture::Llama;
    const bool gemma = config.architecture == Architecture::Gemma3 || config.architecture == Architecture::Gemma4;
    if (gemma) {
        config.sliding_window = file.get<std::uint32_t>(prefix + "attention.sliding_window").value_or(0);
        config.rope_base_local = file.get<float>(prefix + "rope.freq_base_swa").value_or(10000.0f);
        config.rope_scale = 1.0f / file.get<float>(prefix + "rope.scaling.factor").value_or(1.0f);
        config.final_softcap = file.get<float>(prefix + "final_logit_softcapping").value_or(0.0f);
        config.gelu = true;
    }
    if (config.architecture == Architecture::Phi3) {
        config.rope_dim = file.get<std::uint32_t>(prefix + "rope.dimension_count").value_or(0);
        config.rope_attn_factor = file.get<float>(prefix + "rope.scaling.attn_factor").value_or(1.0f);
        config.rope_original_context = file.get<std::uint32_t>(prefix + "rope.scaling.original_context_length").value_or(0);
        if (file.tensor("rope_factors_short.weight") != nullptr) {
            const std::span<const float> fs = f32_tensor(file, "rope_factors_short.weight", config.rope_dim / 2);
            const std::span<const float> fl = f32_tensor(file, "rope_factors_long.weight", config.rope_dim / 2);
            config.rope_factors_short.assign(fs.begin(), fs.end());
            config.rope_factors_long.assign(fl.begin(), fl.end());
            config.rope_factors = config.rope_factors_short;  // a session with a longer context swaps in the long ones
        }
    }
    if (config.architecture == Architecture::Gemma4) {
        if (const auto pattern = file.get<GgufArray>(prefix + "attention.sliding_window_pattern")) {
            // A bool array: one byte per entry.
            if (pattern->elem_type != GgufType::Bool) fail("sliding_window_pattern is not a bool array");
            for (const std::byte b : pattern->data.subspan(0, static_cast<std::size_t>(pattern->count))) {
                config.sliding_pattern.push_back(b != std::byte{0} ? 1 : 0);
            }
        }
        config.head_dim_local = file.get<std::uint32_t>(prefix + "attention.key_length_swa").value_or(0);
        const std::uint32_t shared = file.get<std::uint32_t>(prefix + "attention.shared_kv_layers").value_or(0);
        config.ple_dim = file.get<std::uint32_t>(prefix + "embedding_length_per_layer_input").value_or(0);
        config.attention_scale = 1.0f;
        config.value_norm = true;
        if (const auto ff = file.get<GgufArray>(prefix + "feed_forward_length")) {
            const auto widths = ff->elem_type == GgufType::I32 ? ff->values<std::int32_t>()
                                                                : std::vector<std::int32_t>();
            if (widths.empty()) {
                for (const std::uint32_t w : ff->values<std::uint32_t>()) config.ff_dims.push_back(w);
            } else {
                for (const std::int32_t w : widths) {
                    if (w <= 0) fail("non-positive feed-forward width");
                    config.ff_dims.push_back(static_cast<std::uint32_t>(w));
                }
            }
        }
        if (shared != 0) config.kv_from_start = config.layer_count - shared;  // set below once layer_count is known
    }

    config.embedding_dim = require_u32(file, prefix + "embedding_length");
    config.layer_count = require_u32(file, prefix + "block_count");
    if (config.architecture == Architecture::Gemma4) {
        const std::uint32_t shared = file.get<std::uint32_t>(prefix + "attention.shared_kv_layers").value_or(0);
        if (shared != 0) {
            if (shared + 2 > config.layer_count) fail("more shared K/V layers than layers");
            config.kv_from_start = config.layer_count - shared;
        }
        if (!config.sliding_pattern.empty() && config.sliding_pattern.size() != config.layer_count) fail("sliding window pattern length");
        if (!config.ff_dims.empty() && config.ff_dims.size() != config.layer_count) fail("feed_forward_length array length");
    }
    config.head_count = require_u32(file, prefix + "attention.head_count");
    if (const auto kv = file.get<GgufArray>(prefix + "attention.head_count_kv")) {  // per layer (Gemma 4 12B)
        if (kv->elem_type == GgufType::I32) {
            for (const std::int32_t v : kv->values<std::int32_t>()) config.kv_head_counts.push_back(static_cast<std::uint32_t>(v));
        } else {
            config.kv_head_counts = kv->values<std::uint32_t>();
        }
        if (config.kv_head_counts.size() != config.layer_count) fail("head_count_kv array length");
        config.kv_head_count = *std::max_element(config.kv_head_counts.begin(), config.kv_head_counts.end());
    } else {
        config.kv_head_count = require_u32(file, prefix + "attention.head_count_kv");
    }
    config.feed_forward_dim = config.ff_dims.empty() ? require_u32(file, prefix + "feed_forward_length")
                                                       : *std::max_element(config.ff_dims.begin(), config.ff_dims.end());
    config.context_length = require_u32(file, prefix + "context_length");
    config.rope_base = require_f32(file, prefix + "rope.freq_base");
    config.rms_epsilon = require_f32(file, prefix + "attention.layer_norm_rms_epsilon");

    if (config.embedding_dim == 0 || config.layer_count == 0 || config.head_count == 0 ||
        config.kv_head_count == 0 || config.feed_forward_dim == 0 || config.context_length == 0) {
        fail("a model dimension is zero");
    }
    if (config.architecture == Architecture::LFM2) {
        config.conv_taps = file.get<std::uint32_t>(prefix + "shortconv.l_cache").value_or(0);
        if (config.conv_taps < 2 || config.conv_taps > 8) fail("unsupported short-convolution width");
    }
    for (std::size_t l = 0; l < config.layer_count; ++l) {
        const std::uint32_t kv = config.kv_head_count_of(l);
        if (kv == 0 && config.architecture == Architecture::LFM2) continue;  // a convolution layer
        if (kv == 0 || config.head_count % kv != 0) fail("head_count is not a multiple of head_count_kv");
        if (config.head_count / kv > kMaxHeadsPerKv) fail("more than 16 query heads per key/value head is not supported");
    }

    // Qwen3 sets the head size explicitly; it is not always embedding_dim / head_count.
    config.head_dim = file.get<std::uint32_t>(prefix + "attention.key_length")
                           .value_or(config.embedding_dim / config.head_count);
    if (config.head_dim == 0 || config.head_dim % 32 != 0) fail("head size must be a non-zero multiple of 32");
    if (gemma) config.embedding_scale = std::sqrt(static_cast<float>(config.embedding_dim));
    if (config.head_dim_local != 0 && (config.head_dim_local % 32 != 0 || config.head_dim_local > config.head_dim)) {
        fail("the sliding layers' head size must be a multiple of 32 no larger than the global one");
    }
    if (file.get<std::uint32_t>(prefix + "attention.value_length").value_or(config.head_dim) != config.head_dim) {
        fail("key and value head sizes differ");
    }

    const TensorInfo* embedding = file.tensor("token_embd.weight");
    if (embedding == nullptr || embedding->dims.size() != 2 || embedding->dims[0] != config.embedding_dim) {
        fail("missing or misshapen token_embd.weight");
    }
    if (embedding->dims[1] == 0 || embedding->dims[1] > 0x7FFFFFFF) fail("invalid vocabulary size");
    config.vocab_size = static_cast<std::uint32_t>(embedding->dims[1]);
    return config;
}

Model::Model(const GgufFile& file, bool repack) : repack_(repack) {
    config_ = parse_config(file);
    const std::string prefix = std::string(*file.get<std::string_view>("general.architecture")) + ".";
    const bool gemma = config_.architecture == Architecture::Gemma3 || config_.architecture == Architecture::Gemma4;
    const bool bitnet = config_.architecture == Architecture::BitNet;

    const std::size_t embd = config_.embedding_dim;
    const std::size_t query_dim = static_cast<std::size_t>(config_.head_count) * config_.head_dim;
    const std::size_t kv_dim = static_cast<std::size_t>(config_.kv_head_count) * config_.head_dim;
    const std::size_t ff = config_.feed_forward_dim;

    // Llama 3's rope scaling and Gemma 4's proportional RoPE ship as per-frequency factors.
    if (file.tensor("rope_freqs.weight") != nullptr) {
        const std::span<const float> factors = load(file, "rope_freqs.weight", config_.head_dim / 2);
        config_.rope_factors.assign(factors.begin(), factors.end());
    }
    if (config_.ple_dim != 0) {
        const std::size_t ple_all = static_cast<std::size_t>(config_.ple_dim) * config_.layer_count;
        ple_token_embedding_ = load_matrix(file, "per_layer_token_embd.weight", config_.vocab_size, ple_all);
        ple_model_proj_ = load_matrix(file, "per_layer_model_proj.weight", ple_all, config_.embedding_dim);
        ple_proj_norm_ = load(file, "per_layer_proj_norm.weight", config_.ple_dim);
    }

    token_embedding_ = load_matrix(file, "token_embd.weight", config_.vocab_size, embd);
    // Without a separate output matrix the token embedding is reused for it.
    output_projection_ = file.tensor("output.weight") != nullptr
                             ? load_matrix(file, "output.weight", config_.vocab_size, embd)
                             : token_embedding_;
    output_norm_ = load(file, config_.architecture == Architecture::LFM2 ? "token_embd_norm.weight" : "output_norm.weight", embd);

    layers_.resize(config_.layer_count);
    for (std::size_t i = 0; i < layers_.size(); ++i) {
        const std::string block = "blk." + std::to_string(i) + ".";
        Layer& layer = layers_[i];
        const std::size_t head_dim_l = config_.head_dim_of(i);
        const std::size_t query_dim_l = static_cast<std::size_t>(config_.head_count) * head_dim_l;
        const std::size_t kv_dim_l = static_cast<std::size_t>(config_.kv_head_count_of(i)) * head_dim_l;
        const std::size_t ff_l = config_.ff_dim_of(i);
        const bool own_kv = config_.kv_layer_of(i) == i;
        layer.attention_norm = load(file, block + "attn_norm.weight", embd);
        if (config_.layer_is_conv(i)) {
            layer.conv_in = load_matrix(file, block + "shortconv.in_proj.weight", 3 * embd, embd);
            layer.conv_out = load_matrix(file, block + "shortconv.out_proj.weight", embd, embd);
            layer.conv_weight = load(file, block + "shortconv.conv.weight", std::size_t{config_.conv_taps} * embd);
            layer.feed_forward_norm = load(file, block + "ffn_norm.weight", embd);
            layer.gate = load_matrix(file, block + "ffn_gate.weight", ff_l, embd);
            layer.up = load_matrix(file, block + "ffn_up.weight", ff_l, embd);
            layer.down = load_matrix(file, block + "ffn_down.weight", embd, ff_l);
            continue;
        }
        if (config_.architecture == Architecture::Phi3) {
            // One fused tensor holds Q, K and V rows, another the gate and up rows.
            const std::size_t qkv = query_dim_l + 2 * kv_dim_l;
            layer.query = load_matrix_rows(file, block + "attn_qkv.weight", 0, query_dim_l, qkv, embd);
            layer.key = load_matrix_rows(file, block + "attn_qkv.weight", query_dim_l, kv_dim_l, qkv, embd);
            layer.value = load_matrix_rows(file, block + "attn_qkv.weight", query_dim_l + kv_dim_l, kv_dim_l, qkv, embd);
            layer.attention_output = load_matrix(file, block + "attn_output.weight", embd, query_dim_l);
            layer.feed_forward_norm = load(file, block + "ffn_norm.weight", embd);
            layer.gate = load_matrix_rows(file, block + "ffn_up.weight", 0, ff_l, 2 * ff_l, embd);
            layer.up = load_matrix_rows(file, block + "ffn_up.weight", ff_l, ff_l, 2 * ff_l, embd);
            layer.down = load_matrix(file, block + "ffn_down.weight", embd, ff_l);
            continue;
        }
        layer.query = load_matrix(file, block + "attn_q.weight", query_dim_l, embd);
        if (own_kv) {
            layer.key = load_matrix(file, block + "attn_k.weight", kv_dim_l, embd);
            // Gemma 4 global layers may have no V projection: K serves as V.
            layer.value = file.tensor(block + "attn_v.weight") != nullptr ? load_matrix(file, block + "attn_v.weight", kv_dim_l, embd)
                                                                           : layer.key;
        }
        if (bitnet) {
            layer.attention_sub_norm = load(file, block + "attn_sub_norm.weight", embd);
            layer.feed_forward_sub_norm = load(file, block + "ffn_sub_norm.weight", ff);
        } else if (config_.architecture == Architecture::Qwen3 || config_.architecture == Architecture::LFM2 || gemma) {
            layer.query_norm = load(file, block + "attn_q_norm.weight", head_dim_l);
            if (own_kv) layer.key_norm = load(file, block + "attn_k_norm.weight", head_dim_l);
        }
        if (gemma) {
            layer.attention_post_norm = load(file, block + "post_attention_norm.weight", embd);
            layer.feed_forward_post_norm = load(file, block + "post_ffw_norm.weight", embd);
        }
        if (config_.ple_dim != 0) {
            layer.ple_gate = load_matrix(file, block + "inp_gate.weight", config_.ple_dim, embd);
            layer.ple_proj = load_matrix(file, block + "proj.weight", embd, config_.ple_dim);
            layer.ple_post_norm = load(file, block + "post_norm.weight", embd);
        }
        if (file.tensor(block + "layer_output_scale.weight") != nullptr) {
            layer.out_scale = load(file, block + "layer_output_scale.weight", 1)[0];
        }
        // Projection biases (Qwen2 has them; some Llama-shaped models too). All or none.
        if (file.tensor(block + "attn_q.bias") != nullptr) {
            layer.query_bias = load(file, block + "attn_q.bias", query_dim);
            layer.key_bias = load(file, block + "attn_k.bias", kv_dim);
            layer.value_bias = load(file, block + "attn_v.bias", kv_dim);
        }
        layer.attention_output = load_matrix(file, block + "attn_output.weight", embd, query_dim_l);
        layer.feed_forward_norm = load(file, block + "ffn_norm.weight", embd);
        layer.gate = load_matrix(file, block + "ffn_gate.weight", ff_l, embd);
        layer.up = load_matrix(file, block + "ffn_up.weight", ff_l, embd);
        layer.down = load_matrix(file, block + "ffn_down.weight", embd, ff_l);
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
    return load_matrix_rows(file, name, 0, rows, rows, cols);
}

Matrix Model::load_matrix_rows(const GgufFile& file, std::string_view name, std::size_t row0, std::size_t rows,
                               std::size_t total_rows, std::size_t cols) {
    // GGUF lists the innermost dimension first, so a rows x cols matrix is stored as {cols, rows}.
    const TensorInfo* info = file.tensor(name);
    if (info == nullptr) fail("missing tensor '" + std::string(name) + "'");
    if (info->dims.size() != 2 || info->dims[0] != cols || info->dims[1] != total_rows || row0 + rows > total_rows) {
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
        if (info->data.size() != total_rows * m.row_bytes() || reinterpret_cast<std::uintptr_t>(info->data.data()) % 4 != 0) {
            fail("tensor '" + std::string(name) + "' has unexpected quantised data");
        }
        m.bytes = info->data.subspan(row0 * m.row_bytes(), rows * m.row_bytes());
        if (m.format == WeightFormat::Q4_0 && rows % 4 == 0 && repack_ && kernels::prefers_q4_0x4()) {
            // Interleave four rows per block for the NEON matrix kernels.
            const std::size_t blocks = cols / kBlockSize;
            std::vector<std::byte>& packed = packed_.emplace_back(rows / 4 * blocks * sizeof(BlockQ4_0x4));
            repack_q4_0x4(reinterpret_cast<const BlockQ4_0*>(m.bytes.data()), rows, blocks,
                          reinterpret_cast<BlockQ4_0x4*>(packed.data()));
            m.format = WeightFormat::Q4_0x4;
            m.bytes = packed;
        } else if (m.format == WeightFormat::Q5_0 && rows % 4 == 0 && repack_ && kernels::prefers_q4_0x4()) {
            const std::size_t blocks = cols / kBlockSize;
            std::vector<std::byte>& packed = packed_.emplace_back(rows / 4 * blocks * sizeof(BlockQ5_0x4));
            repack_q5_0x4(reinterpret_cast<const BlockQ5_0*>(m.bytes.data()), rows, blocks,
                          reinterpret_cast<BlockQ5_0x4*>(packed.data()));
            m.format = WeightFormat::Q5_0x4;
            m.bytes = packed;
        } else if (m.format == WeightFormat::TQ2_0 && rows % 4 == 0 && repack_ && kernels::prefers_q4_0x4()) {
            const std::size_t blocks = cols / kSuperBlockSize;
            std::vector<std::byte>& packed = packed_.emplace_back(rows / 4 * blocks * sizeof(BlockTQ2_0x4));
            repack_tq2_0x4(reinterpret_cast<const BlockTQ2_0*>(m.bytes.data()), rows, blocks,
                           reinterpret_cast<BlockTQ2_0x4*>(packed.data()));
            m.format = WeightFormat::TQ2_0x4;
            m.bytes = packed;
        } else if (m.format == WeightFormat::Q4_K && rows % 4 == 0 && repack_ && kernels::prefers_q4_0x4()) {
            const std::size_t blocks = cols / kSuperBlockSize;
            std::vector<std::byte>& packed = packed_.emplace_back(rows / 4 * blocks * sizeof(BlockQ4_Kx4));
            repack_q4_kx4(reinterpret_cast<const BlockQ4_K*>(m.bytes.data()), rows, blocks,
                          reinterpret_cast<BlockQ4_Kx4*>(packed.data()));
            m.format = WeightFormat::Q4_Kx4;
            m.bytes = packed;
        } else if (m.format == WeightFormat::Q6_K && rows % 4 == 0 && repack_ && kernels::prefers_q4_0x4()) {
            const std::size_t blocks = cols / kSuperBlockSize;
            std::vector<std::byte>& packed = packed_.emplace_back(rows / 4 * blocks * sizeof(BlockQ6_Kx4));
            repack_q6_kx4(reinterpret_cast<const BlockQ6_K*>(m.bytes.data()), rows, blocks,
                          reinterpret_cast<BlockQ6_Kx4*>(packed.data()));
            m.format = WeightFormat::Q6_Kx4;
            m.bytes = packed;
        }
    } else {
        m.f32 = load(file, name, total_rows * cols).subspan(row0 * cols, rows * cols);
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

Session::Session(const Model& model, std::size_t threads, std::size_t max_context)
    : model_(model), pool_(std::max<std::size_t>(threads, 1)) {
    const ModelConfig& c = model.config();
    const std::size_t query_dim = static_cast<std::size_t>(c.head_count) * c.head_dim;
    max_context_ = max_context == 0 ? c.context_length : std::min<std::size_t>(max_context, c.context_length);
    key_cache_.resize(c.layer_count);
    value_cache_.resize(c.layer_count);
    kv_staging_.resize(kMaxBatch * static_cast<std::size_t>(c.kv_head_count) * c.head_dim);
    hidden_.resize(kMaxBatch * c.embedding_dim);
    normed_.resize(kMaxBatch * c.embedding_dim);
    query_.resize(kMaxBatch * query_dim);
    attended_.resize(kMaxBatch * query_dim);
    projected_.resize(kMaxBatch * c.embedding_dim);
    gate_.resize(kMaxBatch * c.feed_forward_dim);
    up_.resize(kMaxBatch * c.feed_forward_dim);
    if (c.conv_taps != 0) {
        conv_.resize(kMaxBatch * 3 * c.embedding_dim);
        conv_cache_.resize(c.layer_count);
    }
    rope_factors_ = c.rope_factors;
    if (!c.rope_factors_long.empty() && c.rope_original_context != 0 && max_context_ > c.rope_original_context) {
        rope_factors_ = c.rope_factors_long;  // LongRoPE: the long-context factors past the original context
    }
    rope_.resize(kMaxBatch * c.head_dim);
    rope_local_.resize(kMaxBatch * c.head_dim);
    if (c.ple_dim != 0) {
        ple_.resize(kMaxBatch * static_cast<std::size_t>(c.ple_dim) * c.layer_count);
        ple_gate_.resize(kMaxBatch * c.ple_dim);
    }
    logits_.resize(c.vocab_size);
}

std::span<const float> Session::eval(std::span<const Token> tokens) {
    const ModelConfig& c = model_.config();
    if (tokens.empty()) fail("no tokens to evaluate");
    for (const Token token : tokens) {
        if (token < 0 || static_cast<std::uint32_t>(token) >= c.vocab_size) fail("token id out of range");
    }
    if (tokens.size() > max_context_ - position_) fail("context length exceeded");

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
    if (tokens.size() > max_context_ - position_) fail("context length exceeded");
    eval_batch(tokens, true);
    return std::span<const float>(logits_.data(), tokens.size() * c.vocab_size);
}

void Session::rollback(std::size_t position) {
    if (position > position_) fail("cannot roll forward");
    position_ = position;  // the caches keep their storage; later entries are simply overwritten
}

void Session::eval_batch(std::span<const Token> tokens, bool all_logits) {
    const ModelConfig& c = model_.config();
    const std::size_t n = tokens.size();
    const std::size_t embd = c.embedding_dim;

    const std::size_t length = position_ + n;  // positions visible to the last token of the batch
    // Gemma 4's sliding layers have smaller heads than its global ones; the
    // buffers are sized for the larger (c.head_dim), the tables per kind.
    const std::size_t head_dim_local = c.head_dim_local != 0 ? c.head_dim_local : c.head_dim;

    // Angles are built by repeated multiplication, as llama.cpp does, so the
    // rounding matches. The per-frequency factors (Llama 3's scaling, Gemma
    // 4's proportional RoPE) apply to the global layers only.
    auto build_rope = [&](std::vector<float>& table, std::size_t dim, float base, float scale, bool factors) {
        const float theta_scale = std::pow(base, -2.0f / static_cast<float>(dim));
        for (std::size_t t = 0; t < n; ++t) {
            float theta = static_cast<float>(position_ + t) * scale;
            float* rope = table.data() + t * dim;
            for (std::size_t i = 0; i < dim / 2; ++i) {
                const float angle = factors && !rope_factors_.empty() ? theta / rope_factors_[i] : theta;
                rope[2 * i] = std::cos(angle) * c.rope_attn_factor;
                rope[2 * i + 1] = std::sin(angle) * c.rope_attn_factor;
                theta *= theta_scale;
            }
        }
    };
    build_rope(rope_, c.rope_dim_of(), c.rope_base, c.rope_scale, true);
    if (c.sliding_window != 0) build_rope(rope_local_, head_dim_local, c.rope_base_local, 1.0f, false);

    for (std::size_t t = 0; t < n; ++t) {
        const std::span<float> h(hidden_.data() + t * embd, embd);
        embedding_row(model_.token_embedding(), static_cast<std::size_t>(tokens[t]), h);
        if (c.embedding_scale != 1.0f) {
            for (float& v : h) v *= c.embedding_scale;
        }
    }
    scores_.resize(n * c.head_count * length);

    // Gemma 4's per-layer embedding inputs: for each layer, the projected
    // (and normalised) input embedding plus the token's row of the per-layer
    // table, both scaled, averaged with 1/sqrt(2).
    const std::size_t ple_dim = c.ple_dim, ple_all = ple_dim * c.layer_count;
    if (ple_dim != 0) {
        multiply(model_.ple_model_proj(), hidden_.data(), n, ple_.data());
        const float proj_scale = 1.0f / std::sqrt(static_cast<float>(embd));
        const float table_scale = std::sqrt(static_cast<float>(ple_dim));
        pool_.run(n, [&](std::size_t t) {
            std::vector<float> row(ple_all);
            embedding_row(model_.ple_token_embedding(), static_cast<std::size_t>(tokens[t]), row);
            for (std::size_t l = 0; l < c.layer_count; ++l) {
                const std::span<float> slice(ple_.data() + t * ple_all + l * ple_dim, ple_dim);
                for (float& v : slice) v *= proj_scale;
                rms_norm(slice, model_.ple_proj_norm(), c.rms_epsilon, slice);
                for (std::size_t i = 0; i < ple_dim; ++i) {
                    slice[i] = (slice[i] + row[l * ple_dim + i] * table_scale) * 0.70710678f;
                }
            }
        });
    }

    auto rows = [](std::vector<float>& v, std::size_t t, std::size_t width) {
        return std::span<float>(v.data() + t * width, width);
    };

    for (std::size_t l = 0; l < model_.layers().size(); ++l) {
        const Model::Layer& layer = model_.layers()[l];
        if (c.layer_is_conv(l)) {
            // LFM2 short convolution: [b | c | x] = in_proj(norm(h)); bx = b * x is the
            // conv's state, kept per position; out = c * conv(bx) over the last
            // conv_taps positions (kernel w[d][k], oldest first), then out_proj.
            const std::size_t taps = c.conv_taps;
            std::vector<std::uint16_t>& bx_cache = conv_cache_[l];
            if (bx_cache.empty()) bx_cache.resize(max_context_ * embd);
            {
                const Timed timed(profile_.norm_rope);
                pool_.run(n, [&](std::size_t t) {
                    rms_norm(rows(hidden_, t, embd), layer.attention_norm, c.rms_epsilon, rows(normed_, t, embd));
                });
            }
            multiply(layer.conv_in, normed_.data(), n, conv_.data());
            {
                const Timed timed(profile_.attention);
                pool_.run(n, [&](std::size_t t) {  // the state first, every token's bx before any reads it
                    const float* b = conv_.data() + t * 3 * embd;
                    float* x = conv_.data() + t * 3 * embd + 2 * embd;  // x is overwritten with b * x
                    for (std::size_t d = 0; d < embd; ++d) x[d] *= b[d];
                    kernels::store_f16(x, bx_cache.data() + (position_ + t) * embd, embd);
                });
                pool_.run(n, [&](std::size_t t) {
                    const float* gate = conv_.data() + t * 3 * embd + embd;  // c
                    const std::size_t pos = position_ + t;
                    float* out = attended_.data() + t * embd;
                    std::fill_n(out, embd, 0.0f);
                    std::vector<float> row(embd);
                    for (std::size_t k = 0; k < taps; ++k) {
                        if (pos + k + 1 < taps) continue;  // before the start: zero state
                        kernels::load_f16(bx_cache.data() + (pos + k + 1 - taps) * embd, row.data(), embd);
                        for (std::size_t d = 0; d < embd; ++d) out[d] += layer.conv_weight[d * taps + k] * row[d];
                    }
                    for (std::size_t d = 0; d < embd; ++d) out[d] *= gate[d];
                });
            }
            multiply(layer.conv_out, attended_.data(), n, projected_.data());
        } else {
            const std::size_t head_dim = c.head_dim_of(l);
            const std::size_t query_dim = static_cast<std::size_t>(c.head_count) * head_dim;
            const std::size_t kv_head_count = c.kv_head_count_of(l);
            const std::size_t heads_per_kv = c.head_count / kv_head_count;
            const std::size_t kv_dim = kv_head_count * head_dim;
            const float attention_scale = c.attention_scale != 0.0f ? c.attention_scale : 1.0f / std::sqrt(static_cast<float>(head_dim));
            // A layer without its own K/V reads the cache of an earlier layer of its kind (Gemma 4).
            const std::size_t kv_layer = c.kv_layer_of(l);
            const bool own_kv = kv_layer == l;
            std::vector<std::uint16_t>& keys = key_cache_[kv_layer];
            std::vector<std::uint16_t>& values = value_cache_[kv_layer];
            if (keys.empty()) {
                keys.resize(kv_head_count * max_context_ * head_dim);
                values.resize(keys.size());
            }
            // Row p of key/value head kv in this layer's cache.
            auto cache_row = [&](std::vector<std::uint16_t>& cache, std::size_t kv, std::size_t p) {
                return cache.data() + (kv * max_context_ + p) * head_dim;
            };
            // New keys and values are produced as [token][kv_dim] rows, then copied into the head-major cache.
            float* new_keys = kv_staging_.data();
            float* new_values = projected_.data();  // free until the attention output projection

            {
                const Timed timed(profile_.norm_rope);
                pool_.run(n, [&](std::size_t t) {
                    rms_norm(rows(hidden_, t, embd), layer.attention_norm, c.rms_epsilon, rows(normed_, t, embd));
                });
            }
            if (own_kv) {
                const Target qkv[3] = {{&layer.query, query_.data()}, {&layer.key, new_keys}, {&layer.value, new_values}};
                multiply(qkv, normed_.data(), n);
            } else {
                multiply(layer.query, normed_.data(), n, query_.data());
            }

            Timed* phase = new Timed(profile_.norm_rope);
            const bool slides = c.layer_slides(l);
            pool_.run(n, [&](std::size_t t) {
                // The rotated width: the layer's head size, or the model's partial rotary dimension.
                const std::size_t rope_dim = slides ? head_dim : std::min<std::size_t>(c.rope_dim_of(), head_dim);
                const std::span<const float> rope((slides ? rope_local_ : rope_).data() + t * rope_dim, rope_dim);
                if (!layer.query_bias.empty()) {
                    add_bias({query_.data() + t * query_dim, query_dim}, layer.query_bias);
                    add_bias({new_keys + t * kv_dim, kv_dim}, layer.key_bias);
                    add_bias({new_values + t * kv_dim, kv_dim}, layer.value_bias);
                }
                for (std::size_t h = 0; h < c.head_count; ++h) {
                    const std::span<float> q(query_.data() + t * query_dim + h * head_dim, head_dim);
                    if (!layer.query_norm.empty()) rms_norm(q, layer.query_norm, c.rms_epsilon, q);
                    apply_rope(q, rope, c.rope_neox);
                }
                if (!own_kv) return;
                for (std::size_t h = 0; h < kv_head_count; ++h) {
                    const std::span<float> k(new_keys + t * kv_dim + h * head_dim, head_dim);
                    const std::span<float> v(new_values + t * kv_dim + h * head_dim, head_dim);
                    if (!layer.key_norm.empty()) rms_norm(k, layer.key_norm, c.rms_epsilon, k);
                    if (c.value_norm) rms_norm(v, {}, c.rms_epsilon, v);  // Gemma 4: V normalised, no weight
                    apply_rope(k, rope, c.rope_neox);
                    kernels::store_f16(k.data(), cache_row(keys, h, position_ + t), head_dim);
                    kernels::store_f16(v.data(), cache_row(values, h, position_ + t), head_dim);
                }
            });

            delete phase;
            phase = new Timed(profile_.attention);
            // Causal attention: token t of the batch sees positions 0 to position_ + t.
            // One task covers a key/value head and up to kTokenTile consecutive
            // tokens, so each cached row is read once for all their query heads.
            constexpr std::size_t kTokenTile = 16;  // wide, so the cache is streamed once per 16 tokens, not per 4
            const std::size_t token_tiles = (n + kTokenTile - 1) / kTokenTile;
            pool_.run(token_tiles * kv_head_count, [&](std::size_t task) {
                const std::size_t tile = task / kv_head_count;
                const std::size_t kv = task % kv_head_count;
                const std::size_t t0 = tile * kTokenTile;
                const std::size_t t1 = std::min(n, t0 + kTokenTile);
                const std::size_t visible_last = position_ + t1;  // positions seen by the tile's last token
                // Scores: one row per (token, query head) of the tile.
                float* scores = scores_.data() + (t0 * c.head_count + kv * heads_per_kv) * length;
                auto score_row = [&](std::size_t t, std::size_t g) {
                    return scores + ((t - t0) * c.head_count + g) * length;
                };

                // All the tile's queries (each token's heads for this key/value
                // head) against every position the last token sees; positions a
                // token cannot see are computed too but ignored by its softmax.
                const float* queries[kTokenTile * kMaxHeadsPerKv];
                float* rows_out[kTokenTile * kMaxHeadsPerKv];
                std::size_t nq = 0;
                for (std::size_t t = t0; t < t1; ++t) {
                    for (std::size_t g = 0; g < heads_per_kv; ++g) {
                        queries[nq] = query_.data() + t * query_dim + (kv * heads_per_kv + g) * head_dim;
                        rows_out[nq] = score_row(t, g);
                        ++nq;
                    }
                }
                kernels::scores_f32(queries, nq, cache_row(keys, kv, 0), head_dim, visible_last, head_dim, attention_scale,
                                    rows_out);
                for (std::size_t t = t0; t < t1; ++t) {
                    const std::size_t visible = position_ + t + 1;
                    // A sliding-window layer sees only the last `sliding_window` positions.
                    const std::size_t start = slides && visible > c.sliding_window ? visible - c.sliding_window : 0;
                    for (std::size_t g = 0; g < heads_per_kv; ++g) {
                        kernels::softmax_f32(std::span<float>(score_row(t, g) + start, visible - start));
                    }
                    kernels::weighted_sum_f32(cache_row(values, kv, start), head_dim, visible - start,
                                              score_row(t, 0) + start, length, heads_per_kv, head_dim,
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
        }
        {
            const Timed timed(profile_.norm_rope);
            pool_.run(n, [&](std::size_t t) {
                float* h = hidden_.data() + t * embd;
                float* p = projected_.data() + t * embd;
                if (!layer.attention_post_norm.empty()) {
                    const std::span<float> ps(p, embd);
                    rms_norm(ps, layer.attention_post_norm, c.rms_epsilon, ps);
                }
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
                const std::size_t ff_l = c.ff_dim_of(l);
                float* g = gate_.data() + t * ff_l;
                const float* u = up_.data() + t * ff_l;
                if (c.architecture == Architecture::BitNet) {
                    for (std::size_t i = 0; i < ff_l; ++i) {
                        const float r = std::max(g[i], 0.0f);
                        g[i] = r * r * u[i];
                    }
                } else if (c.gelu) {
                    kernels::gelu_mul_f32(g, u, ff_l);
                } else {
                    kernels::silu_mul_f32(g, u, ff_l);
                }
                if (!layer.feed_forward_sub_norm.empty()) {
                    const std::span<float> gs(g, ff_l);
                    rms_norm(gs, layer.feed_forward_sub_norm, c.rms_epsilon, gs);
                }
            });
        }
        multiply(layer.down, gate_.data(), n, projected_.data());
        {
            const Timed timed(profile_.norm_rope);
            pool_.run(n, [&](std::size_t t) {
                float* h = hidden_.data() + t * embd;
                float* p = projected_.data() + t * embd;
                if (!layer.feed_forward_post_norm.empty()) {
                    const std::span<float> ps(p, embd);
                    rms_norm(ps, layer.feed_forward_post_norm, c.rms_epsilon, ps);
                }
                for (std::size_t i = 0; i < embd; ++i) h[i] += p[i];
            });
        }
        if (ple_dim != 0) {
            // Per-layer embedding: x += norm(proj(gelu(gate x) * ple_input[l])), then the layer's scalar.
            multiply(layer.ple_gate, hidden_.data(), n, ple_gate_.data());
            pool_.run(n, [&](std::size_t t) {
                float* g = ple_gate_.data() + t * ple_dim;
                const float* in = ple_.data() + t * ple_all + l * ple_dim;
                kernels::gelu_mul_f32(g, in, ple_dim);
            });
            multiply(layer.ple_proj, ple_gate_.data(), n, projected_.data());
            pool_.run(n, [&](std::size_t t) {
                float* h = hidden_.data() + t * embd;
                const std::span<float> ps(projected_.data() + t * embd, embd);
                rms_norm(ps, layer.ple_post_norm, c.rms_epsilon, ps);
                for (std::size_t i = 0; i < embd; ++i) h[i] = (h[i] + ps[i]) * layer.out_scale;
            });
        } else if (layer.out_scale != 1.0f) {
            pool_.run(n, [&](std::size_t t) {
                for (float& v : rows(hidden_, t, embd)) v *= layer.out_scale;
            });
        }
        if (std::getenv("BRISK_CPU_DEBUG") != nullptr) {  // the last token's residual after each layer
            double sum = 0.0;
            for (const float v : rows(hidden_, n - 1, embd)) sum += std::fabs(v);
            std::fprintf(stderr, "layer %zu pos %zu: mean|h| %.6f  h[0..3] %.5f %.5f %.5f %.5f\n", l, position_ + n - 1, sum / embd,
                         hidden_[(n - 1) * embd], hidden_[(n - 1) * embd + 1], hidden_[(n - 1) * embd + 2], hidden_[(n - 1) * embd + 3]);
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
    if (c.final_softcap != 0.0f) {
        const std::size_t count = (all_logits ? n : 1) * c.vocab_size;
        for (std::size_t i = 0; i < count; ++i) logits_[i] = c.final_softcap * std::tanh(logits_[i] / c.final_softcap);
    }

    position_ += n;
}

}  // namespace brisk
