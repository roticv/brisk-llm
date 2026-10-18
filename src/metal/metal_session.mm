#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <map>
#include <stdexcept>
#include <string>
#include <unistd.h>

#include "metal_session.h"
#include "quant.h"

namespace brisk {

namespace {

const char* const kShaderSource =
#include "metal_shaders.inc"
    ;

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("metal: " + what); }

// Mirrors of the parameter structs in kernels.metal.
enum Mode : uint32_t { kStore = 0, kResidual = 1, kSiluMul = 2, kReluSqrMul = 3, kGeluMul = 4, kMulGelu = 5 };
struct MatmulParams {
    uint32_t rows, cols, n, mode, write_half;
};
struct GemvParams {
    uint32_t rows[3];
    uint32_t cols, n, mode, paired;
    uint32_t formats[3];
    uint32_t interleave;
};
struct NormParams {
    uint32_t dim;
    float eps;
    uint32_t half_out, residual, then_norm;
    float out_scale;
    uint32_t residual_half;
};
struct RopeParams {
    uint32_t n, position0, head_count, kv_head_count, head_dim, max_context;
    float eps, rope_base, rope_scale;
    uint32_t has_qk_norm, neox, has_bias, has_factors, has_kv, value_norm;
};
struct AttentionParams {
    uint32_t n, position0, head_count, kv_head_count, head_dim, max_context;
    float scale;
    uint32_t write_half, splits, chunk, block_tokens, window, head_blocks;
};
struct PleParams {
    uint32_t n, layers, dim;
    float table_scale;
};
constexpr std::size_t kSmallChunk = 128;  // positions per split of a small batch's attention (as in the kernel)
constexpr std::size_t kSmallBatch = 8;    // rows that take the gemv and the split attention
constexpr std::size_t kTernaryRows = 8;   // row streams per simdgroup of the ternary gemv (as in the kernel)

// The 64x64 tile kernel specialised for a weight format (0 = Q4_0, 1 = TQ2_0).
id<MTLComputePipelineState> tile_pipeline(id<MTLDevice> device, id<MTLLibrary> library, uint32_t format) {
    MTLFunctionConstantValues* constants = [MTLFunctionConstantValues new];
    [constants setConstantValue:&format type:MTLDataTypeUInt atIndex:0];
    NSError* error = nil;
    id<MTLFunction> fn = [library newFunctionWithName:@"matmul_q4_0_mm" constantValues:constants error:&error];
    if (fn == nil) fail(std::string("tile specialisation: ") + error.localizedDescription.UTF8String);
    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&error];
    if (pso == nil) fail(std::string("tile pipeline: ") + error.localizedDescription.UTF8String);
    return pso;
}

// A kernel specialised by one uint function constant.
id<MTLComputePipelineState> specialised_pipeline(id<MTLDevice> device, id<MTLLibrary> library, const char* name,
                                                 NSUInteger index, uint32_t value) {
    MTLFunctionConstantValues* constants = [MTLFunctionConstantValues new];
    [constants setConstantValue:&value type:MTLDataTypeUInt atIndex:index];
    NSError* error = nil;
    id<MTLFunction> fn = [library newFunctionWithName:[NSString stringWithUTF8String:name] constantValues:constants error:&error];
    if (fn == nil) fail(std::string("specialisation of ") + name + ": " + error.localizedDescription.UTF8String);
    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&error];
    if (pso == nil) fail(std::string("pipeline for ") + name + ": " + error.localizedDescription.UTF8String);
    return pso;
}

// The format id the K-quant kernels take at run time (0 for the formats with their own kernels).
uint32_t kquant_format_id(WeightFormat f) {
    switch (f) {
        case WeightFormat::Q4_K: return 2;
        case WeightFormat::Q6_K: return 3;
        case WeightFormat::Q4_1: return 4;
        case WeightFormat::Q5_0: return 5;
        case WeightFormat::Q8_0: return 6;
        default: return 0;
    }
}

// A tensor on the GPU: a buffer and a byte offset into it.
struct Buf {
    id<MTLBuffer> buffer = nil;
    NSUInteger offset = 0;
    WeightFormat format = WeightFormat::Q4_0;  // for matrices
};

constexpr std::size_t kMaxBatch = 512;

}  // namespace

struct MetalSession::Impl {
    const Model& model;
    std::span<const std::byte> file;
    std::size_t max_context = 0;
    std::size_t position = 0;

    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> embed = nil, embed_q4k = nil, embed_q6k = nil, embed_q41 = nil, embed_q50 = nil,
                                embed_q80 = nil, rms_norm = nil, matmul_kq = nil, ple_combine = nil, to_half = nil,
                                attention_by_kind[2] = {nil, nil}, attention_small_by_kind[2] = {nil, nil}, matmul = nil, matmul_tq2 = nil, matmul_q4k = nil,
                                matmul_q6k = nil, matmul_q41 = nil, matmul_mm = nil, matmul_mm_tq2 = nil,
                                matmul_mm_q4k = nil, matmul_mm_q6k = nil, matmul_mm_q41 = nil, matmul_mm_q50 = nil,
                                matmul_mm_q80 = nil, rope = nil,
                                attention = nil,
                                attention_small = nil, attention_combine = nil;
    std::size_t block_tokens = 8;  // tokens per threadgroup of the batched attention (for the largest head size)
    id<MTLBuffer> ple_proj_out = nil, ple_table_out = nil, ple_in = nil, ple_gate_out = nil, ple_gate_h = nil, x_h = nil;  // Gemma 4
    bool tiles_fit = false;  // every matrix has rows and columns in multiples of 64, as the tiled kernel needs

    id<MTLBuffer> weights = nil;  // the whole mapped file, no copy
    std::vector<id<MTLBuffer>> owned;  // tensors that had to be copied (widened norms)

    struct Layer {
        Buf attn_norm, q, k, v, q_norm, k_norm, q_bias, k_bias, v_bias, attn_sub_norm, attn_post_norm, o, ffn_norm, gate, up,
            ffn_sub_norm, ffn_post_norm, down;
        Buf ple_gate, ple_proj, ple_post_norm;  // Gemma 4
        float out_scale = 1.0f;
        std::size_t cache_offset = 0;  // byte offset of this layer's (or its shared source's) K/V cache
        bool own_kv = true;
    };
    std::vector<Layer> layers;
    Buf ple_table, ple_model_proj, ple_proj_norm;  // Gemma 4's per-layer embeddings (the projection quantised to Q8_0)
    std::vector<std::byte> ple_proj_q8;
    Buf token_embd, output, output_norm, rope_factors;

    // Scratch, one row per token of the batch, and the caches.
    // Half copies of the activations the tiled matmul reads, written by their producers.
    id<MTLBuffer> normed_h = nil, attended_h = nil, gate_h = nil;
    id<MTLBuffer> tokens = nil, x = nil, normed = nil, q = nil, k_stage = nil, v_stage = nil, attended = nil,
                  gate = nil, proj = nil, scores = nil, partial = nil, logits = nil, k_cache = nil, v_cache = nil;
    std::vector<float> logits_out;

    Impl(const Model& m, std::span<const std::byte> f, std::size_t max_ctx) : model(m), file(f) {
        const ModelConfig& c = model.config();
        max_context = max_ctx == 0 ? c.context_length : std::min<std::size_t>(max_ctx, c.context_length);
        max_context = (max_context + 7) / 8 * 8;  // the attention kernel reads keys in blocks of 8 positions

        device = MTLCreateSystemDefaultDevice();
        if (device == nil) fail("no Metal device");
        queue = [device newCommandQueue];
        compile();
        map_weights();
        allocate_scratch();
    }

    void compile() {
        NSError* error = nil;
        MTLCompileOptions* options = [MTLCompileOptions new];
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:kShaderSource]
                                                      options:options
                                                        error:&error];
        if (library == nil) fail(std::string("shader compilation failed: ") + error.localizedDescription.UTF8String);
        auto pipeline = [&](const char* name) {
            id<MTLFunction> fn = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
            if (fn == nil) fail(std::string("missing kernel ") + name);
            id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&error];
            if (pso == nil) fail(std::string("pipeline for ") + name + ": " + error.localizedDescription.UTF8String);
            return pso;
        };
        embed = pipeline("embed_q4_0");
        rms_norm = pipeline("rms_norm");
        matmul = pipeline("matmul_q4_0");
        {
            // The ternary gemv references a benchmark-only function constant, so it is built
            // through the specialisation API, with no constants set.
            id<MTLFunction> fn = [library newFunctionWithName:@"matmul_tq2_0"
                                               constantValues:[MTLFunctionConstantValues new]
                                                        error:&error];
            if (fn == nil) fail(std::string("ternary gemv: ") + error.localizedDescription.UTF8String);
            matmul_tq2 = [device newComputePipelineStateWithFunction:fn error:&error];
            if (matmul_tq2 == nil) fail(std::string("ternary gemv pipeline: ") + error.localizedDescription.UTF8String);
        }
        matmul_mm = tile_pipeline(device, library, 0);
        matmul_mm_tq2 = tile_pipeline(device, library, 1);
        matmul_mm_q4k = tile_pipeline(device, library, 2);
        matmul_mm_q6k = tile_pipeline(device, library, 3);
        embed_q4k = specialised_pipeline(device, library, "embed_k", 3, 2);
        embed_q6k = specialised_pipeline(device, library, "embed_k", 3, 3);
        matmul_kq = pipeline("matmul_k");
        matmul_mm_q41 = tile_pipeline(device, library, 4);
        embed_q41 = specialised_pipeline(device, library, "embed_k", 3, 4);
        matmul_mm_q50 = tile_pipeline(device, library, 5);
        matmul_mm_q80 = tile_pipeline(device, library, 6);
        embed_q50 = specialised_pipeline(device, library, "embed_k", 3, 5);
        embed_q80 = specialised_pipeline(device, library, "embed_k", 3, 6);
        rope = pipeline("qk_norm_rope");
        // The attention kernels are specialised per head size; a model with two
        // head sizes (Gemma 4) gets two of each.
        auto dim_class = [](std::uint32_t d) -> uint32_t { return d > 256 ? 512 : d > 128 ? 256 : 128; };
        const ModelConfig& mc = model.config();
        for (int kind = 0; kind < 2; ++kind) {
            const std::uint32_t d = kind == 0 ? mc.head_dim : (mc.head_dim_local != 0 ? mc.head_dim_local : mc.head_dim);
            attention_by_kind[kind] = specialised_pipeline(device, library, "attention", 5, dim_class(d));
            attention_small_by_kind[kind] = specialised_pipeline(device, library, "attention_small", 5, dim_class(d));
        }
        ple_combine = pipeline("ple_combine");
        to_half = pipeline("to_half");
        attention_combine = pipeline("attention_combine");
    }

    // Wraps the mapped file in a buffer without copying: the mapping is
    // page-aligned and the pages past the end of the file read as zeros.
    void map_weights() {
        const auto page = static_cast<std::size_t>(getpagesize());
        const std::size_t length = (file.size() + page - 1) / page * page;
        weights = [device newBufferWithBytesNoCopy:const_cast<std::byte*>(file.data())
                                            length:length
                                           options:MTLResourceStorageModeShared
                                       deallocator:nil];
        if (weights == nil) fail("cannot wrap the model file in a GPU buffer");

        const ModelConfig& c = model.config();
        token_embd = matrix(model.token_embedding(), "token_embd.weight");
        if (c.ple_dim != 0) {
            ple_table = matrix(model.ple_token_embedding(), "per_layer_token_embd.weight");
            ple_proj_norm = floats(model.ple_proj_norm());
            // The projection is BF16 in the file (widened to float by the model); the GPU
            // has no float multiply, so it is quantised to Q8_0 once here.
            const Matrix& mp = model.ple_model_proj();
            if (mp.format != WeightFormat::F32) fail("per_layer_model_proj: unexpected format");
            const std::size_t blocks = mp.cols / 32;
            ple_proj_q8.resize(mp.rows * blocks * sizeof(BlockQ8_0));
            for (std::size_t r = 0; r < mp.rows; ++r) {
                quantize_q8_0(mp.f32_row(r).data(), reinterpret_cast<BlockQ8_0*>(ple_proj_q8.data()) + r * blocks, mp.cols);
            }
            id<MTLBuffer> q8 = [device newBufferWithBytes:ple_proj_q8.data() length:ple_proj_q8.size() options:MTLResourceStorageModeShared];
            owned.push_back(q8);
            ple_model_proj = {q8, 0, WeightFormat::Q8_0};
        }
        if (token_embd.format == WeightFormat::TQ2_0) fail("the Metal backend needs a Q4_0 or K-quant token embedding");
        output = matrix(model.output_projection(), "output.weight");
        output_norm = floats(model.output_norm());
        if (!model.config().rope_factors.empty()) rope_factors = floats(model.config().rope_factors);
        for (std::size_t li = 0; li < model.layers().size(); ++li) {
            const Model::Layer& l = model.layers()[li];
            Layer g;
            g.own_kv = c.kv_layer_of(li) == li;
            g.attn_norm = floats(l.attention_norm);
            g.q = matrix(l.query, "attn_q");
            if (g.own_kv) {
                g.k = matrix(l.key, "attn_k");
                g.v = matrix(l.value, "attn_v");
            }
            if (!l.ple_post_norm.empty()) {
                g.ple_gate = matrix(l.ple_gate, "inp_gate");
                g.ple_proj = matrix(l.ple_proj, "proj");
                g.ple_post_norm = floats(l.ple_post_norm);
            }
            g.out_scale = l.out_scale;
            if (!l.query_norm.empty()) g.q_norm = floats(l.query_norm);
            if (!l.key_norm.empty()) g.k_norm = floats(l.key_norm);
            if (!l.query_bias.empty()) {
                g.q_bias = floats(l.query_bias);
                g.k_bias = floats(l.key_bias);
                g.v_bias = floats(l.value_bias);
            }
            if (!l.attention_sub_norm.empty()) g.attn_sub_norm = floats(l.attention_sub_norm);
            if (!l.attention_post_norm.empty()) g.attn_post_norm = floats(l.attention_post_norm);
            if (!l.feed_forward_post_norm.empty()) g.ffn_post_norm = floats(l.feed_forward_post_norm);
            g.o = matrix(l.attention_output, "attn_output");
            g.ffn_norm = floats(l.feed_forward_norm);
            g.gate = matrix(l.gate, "ffn_gate");
            g.up = matrix(l.up, "ffn_up");
            if (!l.feed_forward_sub_norm.empty()) g.ffn_sub_norm = floats(l.feed_forward_sub_norm);
            g.down = matrix(l.down, "ffn_down");
            layers.push_back(g);
        }
    }

    bool in_file(const void* p, std::size_t bytes) const {
        const auto* b = static_cast<const std::byte*>(p);
        return b >= file.data() && b + bytes <= file.data() + file.size();
    }

    Buf matrix(const Matrix& m, const char* name) {
        const bool supported = m.format == WeightFormat::Q4_0 || m.format == WeightFormat::Q4_1 ||
                               m.format == WeightFormat::Q5_0 || m.format == WeightFormat::Q8_0 ||
                               m.format == WeightFormat::TQ2_0 || m.format == WeightFormat::Q4_K ||
                               m.format == WeightFormat::Q6_K;
        if (!supported || !in_file(m.bytes.data(), m.bytes.size())) {
            fail(std::string("the Metal backend needs ") + name +
                 " as Q4_0, Q4_1, Q5_0, Q8_0, Q4_K, Q6_K or TQ2_0 in the file (load the model without repacking)");
        }
        const bool super_blocks = m.format == WeightFormat::TQ2_0 || m.format == WeightFormat::Q4_K || m.format == WeightFormat::Q6_K;
        if (super_blocks && m.cols % 256 != 0) fail(std::string("columns not a multiple of 256: ") + name);
        return {weights, static_cast<NSUInteger>(m.bytes.data() - file.data()), m.format};
    }

    Buf floats(std::span<const float> v) {
        if (in_file(v.data(), v.size_bytes())) {
            return {weights, static_cast<NSUInteger>(reinterpret_cast<const std::byte*>(v.data()) - file.data())};
        }
        id<MTLBuffer> copy = [device newBufferWithBytes:v.data() length:v.size_bytes() options:MTLResourceStorageModeShared];
        owned.push_back(copy);
        return {copy, 0};
    }

    id<MTLBuffer> make(std::size_t bytes) {
        id<MTLBuffer> b = [device newBufferWithLength:std::max<std::size_t>(bytes, 16) options:MTLResourceStorageModeShared];
        if (b == nil) fail("cannot allocate a GPU buffer of " + std::to_string(bytes) + " bytes");
        return b;
    }

    void allocate_scratch() {
        const ModelConfig& c = model.config();
        const std::size_t embd = c.embedding_dim, qdim = std::size_t{c.head_count} * c.head_dim,
                          kvdim = std::size_t{c.kv_head_count} * c.head_dim, ff = c.feed_forward_dim;
        tokens = make(kMaxBatch * sizeof(int32_t));
        if (c.ple_dim != 0) {
            const std::size_t all = static_cast<std::size_t>(c.ple_dim) * c.layer_count;
            ple_proj_out = make(kMaxBatch * all * 4);
            ple_table_out = make(kMaxBatch * all * 4);
            ple_in = make(kMaxBatch * all * 4);
            ple_gate_out = make(kMaxBatch * c.ple_dim * 4);
            ple_gate_h = make(kMaxBatch * c.ple_dim * 2);
            x_h = make(kMaxBatch * embd * 2);
        }
        x = make(kMaxBatch * embd * 4);
        normed = make(kMaxBatch * embd * 4);
        q = make(kMaxBatch * qdim * 4);
        k_stage = make(kMaxBatch * kvdim * 4);
        v_stage = make(kMaxBatch * kvdim * 4);
        attended = make(kMaxBatch * qdim * 4);
        gate = make(kMaxBatch * ff * 4);
        proj = make(kMaxBatch * embd * 4);  // projections that a post-norm reads before the residual add (Gemma)
        normed_h = make(kMaxBatch * embd * 2);
        attended_h = make(kMaxBatch * qdim * 2);
        gate_h = make(kMaxBatch * ff * 2);
        tiles_fit = embd % 64 == 0 && qdim % 64 == 0 && kvdim % 64 == 0 && ff % 64 == 0 && c.vocab_size % 64 == 0;
        if (c.head_dim > 512 || c.head_dim % 8 != 0) fail("attention kernel supports head_dim <= 512, a multiple of 8");
        // The batched kernel holds 16 query rows (8 at 512 wide): tokens per threadgroup
        // = rows / (query heads per key/value head), at least 1; the small kernel takes
        // 8 / (head size / 128) heads per threadgroup and runs several per token.
        block_tokens = std::max<std::size_t>(1, 16 / (c.head_count / c.kv_head_count));
        const std::size_t block_tokens_min = std::max<std::size_t>(1, (c.head_dim > 256 ? 8 : 16) / (c.head_count / c.kv_head_count));
        // 16 score rows per attention threadgroup: (batch / block_tokens) x kv heads threadgroups.
        scores = make(((kMaxBatch + block_tokens_min - 1) / block_tokens_min) * c.kv_head_count * 16 * max_context * 4);
        // Partials: kSmallBatch tokens x heads x splits, each head_dim + 2 floats.
        partial = make(kSmallBatch * c.head_count * ((max_context + kSmallChunk - 1) / kSmallChunk) * (c.head_dim + 2) * 4);
        logits = make(Engine::kMaxScoredTokens * c.vocab_size * 4);
        // One K/V region per layer that has its own cache, sized by that layer's head size.
        std::size_t cache = 0;
        for (std::size_t l = 0; l < layers.size(); ++l) {
            if (layers[l].own_kv) {
                layers[l].cache_offset = cache;
                cache += std::size_t{c.kv_head_count} * max_context * c.head_dim_of(l) * 4;
            }
        }
        for (std::size_t l = 0; l < layers.size(); ++l) {
            if (!layers[l].own_kv) layers[l].cache_offset = layers[c.kv_layer_of(l)].cache_offset;
        }
        k_cache = make(cache);
        v_cache = make(cache);
        // The attention kernel multiplies whole 8-position blocks, so cache rows past
        // the sequence are read (with zero weights): they must hold finite values.
        std::memset(k_cache.contents, 0, cache);
        std::memset(v_cache.contents, 0, cache);
    }

    // ---- encoding helpers ----

    static void set(id<MTLComputeCommandEncoder> enc, const Buf& b, NSUInteger index) {
        [enc setBuffer:b.buffer offset:b.offset atIndex:index];
    }
    static void set(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> b, NSUInteger index) {
        [enc setBuffer:b offset:0 atIndex:index];
    }
    template <typename T>
    static void bytes(id<MTLComputeCommandEncoder> enc, const T& v, NSUInteger index) {
        [enc setBytes:&v length:sizeof(T) atIndex:index];
    }
    static void groups(id<MTLComputeCommandEncoder> enc, NSUInteger count, NSUInteger size) {
        [enc dispatchThreadgroups:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(size, 1, 1)];
    }

    // An activation matrix: the float rows, and the half copy the tiled
    // matmul reads (written by whichever kernel produced the activation).
    struct Act {
        id<MTLBuffer> f = nil;
        id<MTLBuffer> h = nil;
    };

    bool tiled(std::size_t n) const { return tiles_fit && n > kSmallBatch; }

    id<MTLComputePipelineState> tile_for(WeightFormat f) const {
        switch (f) {
            case WeightFormat::TQ2_0: return matmul_mm_tq2;
            case WeightFormat::Q4_K: return matmul_mm_q4k;
            case WeightFormat::Q6_K: return matmul_mm_q6k;
            case WeightFormat::Q4_1: return matmul_mm_q41;
            case WeightFormat::Q5_0: return matmul_mm_q50;
            case WeightFormat::Q8_0: return matmul_mm_q80;
            default: return matmul_mm;
        }
    }
    id<MTLComputePipelineState> gemv_for(WeightFormat f) const {
        switch (f) {
            case WeightFormat::TQ2_0: return matmul_tq2;
            case WeightFormat::Q4_K:
            case WeightFormat::Q6_K:
            case WeightFormat::Q4_1:
            case WeightFormat::Q5_0:
            case WeightFormat::Q8_0: return matmul_kq;
            default: return matmul;
        }
    }

    void encode_norm(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> in, const Buf& w, Act out, std::size_t rows,
                     std::size_t dim, NSUInteger in_offset = 0, NSUInteger out_offset = 0) {
        [enc setComputePipelineState:rms_norm];
        [enc setBuffer:in offset:in_offset atIndex:0];
        set(enc, w, 1);
        [enc setBuffer:out.f offset:out_offset atIndex:2];
        const NormParams p{static_cast<uint32_t>(dim), model.config().rms_epsilon, tiled(rows) ? 1u : 0u, 0, 0, 1.0f, 0};
        bytes(enc, p, 3);
        [enc setBuffer:out.h offset:out_offset / 2 atIndex:4];
        [enc setBuffer:out.f offset:0 atIndex:5];
        set(enc, w, 6);
        groups(enc, rows, 256);
    }

    // residual += rms_norm(in) * w (Gemma's post-norms), and when `next` is
    // given, out = rms_norm(residual) * next in the same pass (the following
    // pre-norm).
    void encode_norm_add(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> in, const Buf& w, id<MTLBuffer> residual,
                         std::size_t rows, std::size_t dim, const Buf* next = nullptr, Act out = Act{nil, nil},
                         float out_scale = 1.0f, id<MTLBuffer> residual_h = nil) {
        [enc setComputePipelineState:rms_norm];
        set(enc, in, 0);
        set(enc, w, 1);
        set(enc, next != nullptr ? out.f : in, 2);
        const NormParams p{static_cast<uint32_t>(dim), model.config().rms_epsilon, next != nullptr && tiled(rows) ? 1u : 0u, 1,
                           next != nullptr ? 1u : 0u, out_scale, residual_h != nil && tiled(rows) ? 1u : 0u};
        bytes(enc, p, 3);
        set(enc, next != nullptr && out.h != nil ? out.h : in, 4);
        set(enc, residual, 5);
        set(enc, next != nullptr ? *next : w, 6);
        set(enc, residual_h != nil ? residual_h : in, 7);
        groups(enc, rows, 256);
    }

    struct Target {
        const Buf* w;
        std::size_t rows;
        Act out;
    };

    // Multiplies up to three matrices (same column count) by the same `n`
    // input rows, with `mode` applied to the result (aux is its other
    // operand). Few rows: one fused gemv dispatch. Many: the tiled
    // simdgroup-matrix multiply per matrix. The result's half copy is written
    // when the target has one.
    // gate and up projections with the activation applied: out = f(gate) * up.
    // The gemv pairs the rows in one dispatch; the tiled path runs gate, then
    // up with the epilogue reading it.
    void encode_gated(id<MTLComputeCommandEncoder> enc, const Buf& gate_w, const Buf& up_w, std::size_t rows,
                      std::size_t cols, Act in, std::size_t n, Mode mode, Act out) {
        if (tiled(n) || gate_w.format != up_w.format) {
            encode_matmul(enc, {Target{&gate_w, rows, Act{out.f, nil}}}, cols, in, n);
            encode_matmul(enc, {Target{&up_w, rows, out}}, cols, in, n, mode, out.f);
            return;
        }
        const bool ternary = gate_w.format == WeightFormat::TQ2_0;
        [enc setComputePipelineState:gemv_for(gate_w.format)];
        set(enc, gate_w, 0);
        set(enc, up_w, 1);
        set(enc, up_w, 2);
        set(enc, in.f, 3);
        for (NSUInteger i = 4; i < 7; ++i) set(enc, out.f, i);
        const uint32_t fid = kquant_format_id(gate_w.format);
        const GemvParams p{{static_cast<uint32_t>(rows), 0, 0}, static_cast<uint32_t>(cols), static_cast<uint32_t>(n), mode, 1,
                           {fid, fid, fid}, (cols / 32) % 32 != 0 ? 1u : 0u};
        bytes(enc, p, 7);
        set(enc, in.f, 8);
        // Ternary and Q4_0: groups of (rows per simdgroup / 2) rows of both matrices, 2 simdgroups per threadgroup.
        if (ternary) groups(enc, ((rows + kTernaryRows / 2 - 1) / (kTernaryRows / 2) * n + 1) / 2, 64);
        else if (gate_w.format == WeightFormat::Q4_0) groups(enc, ((rows + 1) / 2 * n + 1) / 2, 64);
        else groups(enc, (rows * n + 3) / 4, 128);
    }

    id<MTLComputePipelineState> embed_for(WeightFormat f) const {
        switch (f) {
            case WeightFormat::Q4_K: return embed_q4k;
            case WeightFormat::Q6_K: return embed_q6k;
            case WeightFormat::Q4_1: return embed_q41;
            case WeightFormat::Q5_0: return embed_q50;
            case WeightFormat::Q8_0: return embed_q80;
            default: return embed;
        }
    }

    // `aux` is the epilogue's second operand (a residual or a gate), read at `aux_offset` bytes.
    void encode_matmul(id<MTLComputeCommandEncoder> enc, std::initializer_list<Target> targets, std::size_t cols,
                       Act in, std::size_t n, Mode mode = kStore, id<MTLBuffer> aux = nil, std::size_t aux_offset = 0) {
        if (tiled(n)) {
            for (const Target& t : targets) {
                [enc setComputePipelineState:tile_for(t.w->format)];
                set(enc, *t.w, 0);
                set(enc, in.h, 1);
                set(enc, t.out.f, 2);
                const MatmulParams p{static_cast<uint32_t>(t.rows), static_cast<uint32_t>(cols), static_cast<uint32_t>(n),
                                     mode, t.out.h != nil ? 1u : 0u};
                bytes(enc, p, 3);
                [enc setBuffer:aux != nil ? aux : t.out.f offset:aux_offset atIndex:4];
                set(enc, t.out.h != nil ? t.out.h : in.h, 5);
                [enc dispatchThreadgroups:MTLSizeMake(t.rows / 64, (n + 63) / 64, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }
            return;
        }
        // One dispatch per kernel: the K-quant kernel takes a format per matrix
        // (a Q4_K_M file mixes Q4_K and Q6_K in a layer), the others one format.
        const WeightFormat format = targets.begin()->w->format;
        bool mixed = false;
        for (const Target& t : targets) mixed |= gemv_for(t.w->format) != gemv_for(format);
        if (mixed) {
            for (const Target& t : targets) encode_matmul(enc, {t}, cols, in, n, mode, aux, aux_offset);
            return;
        }
        const bool ternary = format == WeightFormat::TQ2_0;
        [enc setComputePipelineState:gemv_for(format)];
        GemvParams p{{0, 0, 0}, static_cast<uint32_t>(cols), static_cast<uint32_t>(n), mode, 0, {0, 0, 0},
                     (cols / 32) % 32 != 0 ? 1u : 0u};
        const bool q4 = format == WeightFormat::Q4_0;
        std::size_t pairs = 0, i = 0;  // row groups: kTernaryRows for TQ2_0, 4 for Q4_0, pairs for the K-quants
        for (const Target& t : targets) {
            set(enc, *t.w, i);
            set(enc, t.out.f, 4 + i);
            p.rows[i] = static_cast<uint32_t>(t.rows);
            p.formats[i] = kquant_format_id(t.w->format);
            pairs += ternary ? (t.rows + kTernaryRows - 1) / kTernaryRows : q4 ? (t.rows + 3) / 4 : (t.rows + 1) / 2;
            ++i;
        }
        for (; i < 3; ++i) {  // unused slots: bind something valid
            set(enc, *targets.begin()->w, i);
            set(enc, targets.begin()->out.f, 4 + i);
        }
        set(enc, in.f, 3);
        bytes(enc, p, 7);
        [enc setBuffer:aux != nil ? aux : in.f offset:aux_offset atIndex:8];
        if (ternary || q4) groups(enc, (pairs * n + 1) / 2, 64);
        else groups(enc, (pairs * n + 3) / 4, 128);
    }

    // ---- the forward pass ----

    // With BRISK_GPU_PROFILE set, each kernel class is committed and timed
    // separately (slower overall; for finding what to tune).
    const bool profiling = std::getenv("BRISK_GPU_PROFILE") != nullptr;
    const bool timing = std::getenv("BRISK_GPU_TIME") != nullptr;  // GPU vs wall time per batch
    std::map<std::string, double> profile;

    void eval_batch(std::span<const Token> toks, bool all_logits) {
        const ModelConfig& c = model.config();
        const std::size_t n = toks.size();
        const std::size_t embd = c.embedding_dim;
        const bool bitnet = c.architecture == Architecture::BitNet;

        auto* ids = static_cast<int32_t*>(tokens.contents);
        for (std::size_t t = 0; t < n; ++t) ids[t] = toks[t];
        const Act normed_a{normed, normed_h}, attended_a{attended, attended_h}, gate_a{gate, gate_h};
        const Act x_a{x, nil}, q_a{q, nil}, k_a{k_stage, nil}, v_a{v_stage, nil}, logits_a{logits, nil};

        @autoreleasepool {
            const auto encode_start = std::chrono::steady_clock::now();
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            // In profile mode, ends the current command buffer, waits, and charges its time to `name`.
            auto checkpoint = [&](const char* name) {
                if (!profiling) return;
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                profile[name] += cb.GPUEndTime - cb.GPUStartTime;
                cb = [queue commandBuffer];
                enc = [cb computeCommandEncoder];
            };

            [enc setComputePipelineState:embed_for(token_embd.format)];
            set(enc, token_embd, 0);
            set(enc, tokens, 1);
            set(enc, x, 2);
            const uint32_t embd32 = static_cast<uint32_t>(embd);
            bytes(enc, embd32, 3);
            bytes(enc, c.embedding_scale, 4);
            [enc dispatchThreads:MTLSizeMake(embd / 32, n, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

            if (c.ple_dim != 0) {
                // Per-layer embedding inputs (Gemma 4): the projected input embedding, scaled and
                // normalised per layer slice, plus the token's table row, averaged.
                const std::size_t all = static_cast<std::size_t>(c.ple_dim) * c.layer_count;
                if (tiled(n)) {  // the tile path reads halves: convert the embeddings
                    [enc setComputePipelineState:to_half];
                    set(enc, x, 0);
                    set(enc, x_h, 1);
                    const uint32_t count = static_cast<uint32_t>(n * embd);
                    bytes(enc, count, 2);
                    [enc dispatchThreads:MTLSizeMake(n * embd, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                }
                encode_matmul(enc, {Target{&ple_model_proj, all, Act{ple_proj_out, nil}}}, embd, Act{x, x_h}, n);
                {
                    // x = proj / sqrt(embd), then rms_norm per 256-slice with ple_proj_norm, in place.
                    [enc setComputePipelineState:rms_norm];
                    set(enc, ple_proj_out, 0);
                    set(enc, ple_proj_norm, 1);
                    set(enc, ple_proj_out, 2);
                    // The 1/sqrt(embd) scale cancels in the normalisation (rms_norm is scale-invariant up to eps).
                    const NormParams np{c.ple_dim, c.rms_epsilon * static_cast<float>(embd), 0, 0, 0, 1.0f, 0};
                    bytes(enc, np, 3);
                    set(enc, ple_proj_out, 4);
                    set(enc, ple_proj_out, 5);
                    set(enc, ple_proj_norm, 6);
                    groups(enc, n * c.layer_count, 256);
                }
                {
                    [enc setComputePipelineState:embed_for(ple_table.format)];
                    set(enc, ple_table, 0);
                    set(enc, tokens, 1);
                    set(enc, ple_table_out, 2);
                    const uint32_t all32 = static_cast<uint32_t>(all);
                    bytes(enc, all32, 3);
                    const float one = 1.0f;
                    bytes(enc, one, 4);
                    [enc dispatchThreads:MTLSizeMake(all / 32, n, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                }
                [enc setComputePipelineState:ple_combine];
                set(enc, ple_proj_out, 0);
                set(enc, ple_table_out, 1);
                set(enc, ple_in, 2);
                const PleParams pp{static_cast<uint32_t>(n), c.layer_count, c.ple_dim, std::sqrt(static_cast<float>(c.ple_dim))};
                bytes(enc, pp, 3);
                [enc dispatchThreads:MTLSizeMake(n * all, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            checkpoint("embed");
            // BRISK_GPU_LAYERS caps the layers evaluated (debugging: the logits are then meaningless).
            static const std::size_t layer_cap = [] {
                const char* env = std::getenv("BRISK_GPU_LAYERS");
                return env != nullptr ? static_cast<std::size_t>(std::atoi(env)) : std::size_t(-1);
            }();
            bool next_prenormed = false;  // the previous layer's fused post-norm already wrote this layer's pre-norm
            for (std::size_t l = 0; l < std::min(layers.size(), layer_cap); ++l) {
                const Layer& g = layers[l];
                const std::size_t head_dim = c.head_dim_of(l);
                const std::size_t qdim = std::size_t{c.head_count} * head_dim, kvdim = std::size_t{c.kv_head_count} * head_dim;
                const std::size_t ff = c.ff_dim_of(l);
                const std::size_t cache_layer = g.cache_offset;
                const int kind = c.head_dim_of(l) == c.head_dim ? 0 : 1;  // attention pipelines by head size
                const std::size_t rows_max = head_dim > 256 ? 8 : 16;
                const std::size_t block_tokens_l = std::max<std::size_t>(1, rows_max / (c.head_count / c.kv_head_count));

                if (!next_prenormed) encode_norm(enc, x, g.attn_norm, normed_a, n, embd);
                next_prenormed = false;
                checkpoint("norm");
                if (g.own_kv) {
                    encode_matmul(enc, {Target{&g.q, qdim, q_a}, Target{&g.k, kvdim, k_a}, Target{&g.v, kvdim, v_a}}, embd,
                                  normed_a, n);
                } else {
                    encode_matmul(enc, {Target{&g.q, qdim, q_a}}, embd, normed_a, n);
                }
                checkpoint("matmul");

                [enc setComputePipelineState:rope];
                set(enc, q, 0);
                set(enc, k_stage, 1);
                set(enc, v_stage, 2);
                set(enc, g.q_norm.buffer != nil ? g.q_norm : g.attn_norm, 3);  // unused when has_qk_norm is 0
                set(enc, g.k_norm.buffer != nil ? g.k_norm : g.attn_norm, 4);
                [enc setBuffer:k_cache offset:cache_layer atIndex:5];
                [enc setBuffer:v_cache offset:cache_layer atIndex:6];
                const bool slides = c.layer_slides(l);
                // The per-frequency factors apply to the global layers only.
                const RopeParams rp{static_cast<uint32_t>(n), static_cast<uint32_t>(position), c.head_count, c.kv_head_count,
                                    static_cast<uint32_t>(head_dim), static_cast<uint32_t>(max_context), c.rms_epsilon,
                                    slides ? c.rope_base_local : c.rope_base, slides ? 1.0f : c.rope_scale,
                                    g.q_norm.buffer != nil ? 1u : 0u, c.rope_neox ? 1u : 0u,
                                    g.q_bias.buffer != nil ? 1u : 0u, rope_factors.buffer != nil && !slides ? 1u : 0u,
                                    g.own_kv ? 1u : 0u, c.value_norm ? 1u : 0u};
                bytes(enc, rp, 7);
                set(enc, g.q_bias.buffer != nil ? g.q_bias : g.attn_norm, 8);  // unused when has_bias is 0
                set(enc, g.k_bias.buffer != nil ? g.k_bias : g.attn_norm, 9);
                set(enc, g.v_bias.buffer != nil ? g.v_bias : g.attn_norm, 10);
                set(enc, rope_factors.buffer != nil ? rope_factors : g.attn_norm, 11);
                set(enc, v_stage, 12);
                groups(enc, (n * (c.head_count + (g.own_kv ? c.kv_head_count : 0)) + 3) / 4, 128);
                checkpoint("rope");

                // A small batch has too few (token, head) items to fill the GPU, so
                // the small kernel spreads each one's positions over splits of
                // kSmallChunk that a second kernel merges. Batches use the tiled kernel.
                const std::size_t visible = position + n;
                const bool small = n <= kSmallBatch;
                uint32_t splits = 1, chunk = static_cast<uint32_t>((max_context + 7) / 8 * 8);
                if (small) {
                    splits = static_cast<uint32_t>((visible + kSmallChunk - 1) / kSmallChunk);
                    chunk = static_cast<uint32_t>(kSmallChunk);
                }
                // BitNet normalises the attention output first, which rewrites the half copy.
                const std::size_t group = c.head_count / c.kv_head_count;
                const std::size_t max_group = 8 / (head_dim > 256 ? 4 : head_dim > 128 ? 2 : 1);  // heads per small-kernel threadgroup
                const std::size_t head_blocks = (group + max_group - 1) / max_group;
                const AttentionParams ap{static_cast<uint32_t>(n), static_cast<uint32_t>(position), c.head_count,
                                         c.kv_head_count, static_cast<uint32_t>(head_dim), static_cast<uint32_t>(max_context),
                                         c.attention_scale != 0.0f ? c.attention_scale : 1.0f / std::sqrt(static_cast<float>(head_dim)),
                                         tiled(n) && !bitnet ? 1u : 0u, splits, chunk, static_cast<uint32_t>(block_tokens_l),
                                         slides ? c.sliding_window : 0u, static_cast<uint32_t>(head_blocks)};
                if (small) {
                    [enc setComputePipelineState:attention_small_by_kind[kind]];
                    set(enc, q, 0);
                    [enc setBuffer:k_cache offset:cache_layer atIndex:1];
                    [enc setBuffer:v_cache offset:cache_layer atIndex:2];
                    set(enc, partial, 3);
                    bytes(enc, ap, 4);
                    groups(enc, n * c.kv_head_count * head_blocks * splits, 128);
                } else {
                    [enc setComputePipelineState:attention_by_kind[kind]];
                    set(enc, q, 0);
                    [enc setBuffer:k_cache offset:cache_layer atIndex:1];
                    [enc setBuffer:v_cache offset:cache_layer atIndex:2];
                    set(enc, attended, 3);
                    set(enc, scores, 4);
                    bytes(enc, ap, 5);
                    set(enc, attended_h, 6);
                    groups(enc, ((n + block_tokens_l - 1) / block_tokens_l) * c.kv_head_count, 128);
                }
                if (small) {
                    [enc setComputePipelineState:attention_combine];
                    set(enc, partial, 0);
                    set(enc, attended, 1);
                    bytes(enc, ap, 2);
                    set(enc, attended_h, 3);
                    groups(enc, (n * c.head_count + 3) / 4, 128);
                }
                checkpoint("attention");

                if (bitnet) encode_norm(enc, attended, g.attn_sub_norm, attended_a, n, qdim);
                checkpoint("norm");
                if (g.attn_post_norm.buffer != nil) {
                    // Gemma: x += post_norm(o(attended)), then the FFN's pre-norm in the same pass.
                    encode_matmul(enc, {Target{&g.o, embd, Act{proj, nil}}}, qdim, attended_a, n);
                    encode_norm_add(enc, proj, g.attn_post_norm, x, n, embd, &g.ffn_norm, normed_a);
                } else {
                    encode_matmul(enc, {Target{&g.o, embd, x_a}}, qdim, attended_a, n, kResidual, x);
                    encode_norm(enc, x, g.ffn_norm, normed_a, n, embd);
                }
                checkpoint("matmul");
                // gate, then up with the activation applied against it; BitNet
                // normalises the product afterwards (that norm writes the half copy).
                encode_gated(enc, g.gate, g.up, ff, embd, normed_a, n, bitnet ? kReluSqrMul : c.gelu ? kGeluMul : kSiluMul,
                             bitnet ? Act{gate, nil} : gate_a);
                if (bitnet) encode_norm(enc, gate, g.ffn_sub_norm, gate_a, n, ff);
                checkpoint("matmul");
                if (g.ple_gate.buffer != nil) {
                    // Gemma 4: x += post_ffn_norm(down(...)); then x += norm(proj(gelu(gate x) * ple[l])), scaled.
                    encode_matmul(enc, {Target{&g.down, embd, Act{proj, nil}}}, ff, gate_a, n);
                    // The gate reads x itself (its half copy, x_h, written here for the tile path).
                    encode_norm_add(enc, proj, g.ffn_post_norm, x, n, embd, nullptr, Act{nil, nil}, 1.0f, x_h);
                    encode_matmul(enc, {Target{&g.ple_gate, c.ple_dim, Act{ple_gate_out, ple_gate_h}}}, embd, Act{x, x_h}, n,
                                  kMulGelu, ple_in, l * n * c.ple_dim * 4);
                    encode_matmul(enc, {Target{&g.ple_proj, embd, Act{proj, nil}}}, c.ple_dim, Act{ple_gate_out, ple_gate_h}, n);
                    const bool last = l + 1 == layers.size();
                    encode_norm_add(enc, proj, g.ple_post_norm, x, n, embd, last ? nullptr : &layers[l + 1].attn_norm, normed_a,
                                    g.out_scale);
                    next_prenormed = !last;
                } else if (g.ffn_post_norm.buffer != nil) {
                    encode_matmul(enc, {Target{&g.down, embd, Act{proj, nil}}}, ff, gate_a, n);
                    // ... and the next layer's attention pre-norm in the same pass.
                    const bool last = l + 1 == layers.size();
                    encode_norm_add(enc, proj, g.ffn_post_norm, x, n, embd, last ? nullptr : &layers[l + 1].attn_norm, normed_a);
                    next_prenormed = !last;
                } else {
                    encode_matmul(enc, {Target{&g.down, embd, x_a}}, ff, gate_a, n, kResidual, x);
                }
                checkpoint("matmul");
            }

            if (all_logits) {
                encode_norm(enc, x, output_norm, normed_a, n, embd);
                encode_matmul(enc, {Target{&output, c.vocab_size, logits_a}}, embd, normed_a, n);
            } else {
                encode_norm(enc, x, output_norm, normed_a, 1, embd, (n - 1) * embd * 4, 0);
                encode_matmul(enc, {Target{&output, c.vocab_size, logits_a}}, embd, normed_a, 1);
            }
            checkpoint("output");
            [enc endEncoding];
            const auto commit_time = std::chrono::steady_clock::now();
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.error != nil) fail(std::string("command buffer failed: ") + cb.error.localizedDescription.UTF8String);
            if (timing) {
                const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - commit_time).count();
                const double encode = std::chrono::duration<double>(commit_time - encode_start).count();
                std::fprintf(stderr, "gpu batch of %zu: encode %.4f s, gpu %.4f s, commit-to-done %.4f s\n", n, encode,
                             cb.GPUEndTime - cb.GPUStartTime, wall);
            }
        }
        if (profiling) {
            std::string line = "gpu profile (s):";
            for (const auto& [name, secs] : profile) line += " " + name + " " + std::to_string(secs).substr(0, 5);
            std::fprintf(stderr, "%s\n", line.c_str());
        }

        if (std::getenv("BRISK_GPU_DEBUG") != nullptr) {
            auto report = [&](const char* name, id<MTLBuffer> b, std::size_t count) {
                const float* v = static_cast<const float*>(b.contents);
                float worst = 0.0f;
                std::size_t nans = 0;
                for (std::size_t i = 0; i < count; ++i) {
                    if (std::isnan(v[i]) || std::isinf(v[i])) ++nans;
                    else worst = std::max(worst, std::fabs(v[i]));
                }
                std::fprintf(stderr, "  %-9s max %12.4f  non-finite %zu / %zu\n", name, worst, nans, count);
            };
            report("x", x, n * embd);
            report("normed", normed, n * embd);
            const std::size_t qdim = std::size_t{c.head_count} * c.head_dim, kvdim = std::size_t{c.kv_head_count} * c.head_dim,
                              ff = c.feed_forward_dim;
            report("q", q, n * qdim);
            report("k_stage", k_stage, n * kvdim);
            report("attended", attended, n * qdim);
            report("proj", proj, n * embd);
            report("gate", gate, n * ff);

        }
        const std::size_t count = (all_logits ? n : 1) * c.vocab_size;
        logits_out.resize(count);
        std::memcpy(logits_out.data(), logits.contents, count * sizeof(float));
        if (c.final_softcap != 0.0f) {
            for (float& v : logits_out) v = c.final_softcap * std::tanh(v / c.final_softcap);
        }
        position += n;
    }
};

MetalSession::MetalSession(const Model& model, std::span<const std::byte> file, std::size_t max_context)
    : impl_(std::make_unique<Impl>(model, file, max_context)) {}

MetalSession::~MetalSession() = default;

std::span<const float> MetalSession::eval(std::span<const Token> tokens) {
    const ModelConfig& c = impl_->model.config();
    if (tokens.empty()) fail("no tokens to evaluate");
    for (const Token token : tokens) {
        if (token < 0 || static_cast<std::uint32_t>(token) >= c.vocab_size) fail("token id out of range");
    }
    if (tokens.size() > impl_->max_context - impl_->position) fail("context length exceeded");
    for (std::size_t begin = 0; begin < tokens.size(); begin += kMaxBatch) {
        impl_->eval_batch(tokens.subspan(begin, std::min(kMaxBatch, tokens.size() - begin)), false);
    }
    return impl_->logits_out;
}

std::span<const float> MetalSession::eval_all(std::span<const Token> tokens) {
    const ModelConfig& c = impl_->model.config();
    if (tokens.empty() || tokens.size() > kMaxScoredTokens) fail("eval_all takes 1 to 64 tokens");
    for (const Token token : tokens) {
        if (token < 0 || static_cast<std::uint32_t>(token) >= c.vocab_size) fail("token id out of range");
    }
    if (tokens.size() > impl_->max_context - impl_->position) fail("context length exceeded");
    impl_->eval_batch(tokens, true);
    return impl_->logits_out;
}

void MetalSession::rollback(std::size_t position) {
    if (position > impl_->position) fail("cannot roll forward");
    impl_->position = position;
}

std::size_t MetalSession::position() const { return impl_->position; }

namespace {

// Everything the standalone matmul entry points share: a device, the
// pipelines, and the encoding of one multiply.
struct StandaloneMatmul {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;
    id<MTLComputePipelineState> gemv = nil, mm = nil;
    uint32_t format = 0;  // the K-quant format id, when selected

    StandaloneMatmul() {
        device = MTLCreateSystemDefaultDevice();
        if (device == nil) fail("no Metal device");
        queue = [device newCommandQueue];
        NSError* error = nil;
        library = [device newLibraryWithSource:[NSString stringWithUTF8String:kShaderSource]
                                                      options:[MTLCompileOptions new]
                                                        error:&error];
        if (library == nil) fail(std::string("shader compilation failed: ") + error.localizedDescription.UTF8String);
        auto pipeline = [&](const char* name) {
            id<MTLComputePipelineState> pso =
                [device newComputePipelineStateWithFunction:[library newFunctionWithName:[NSString stringWithUTF8String:name]]
                                                      error:&error];
            if (pso == nil) fail(std::string("pipeline for ") + name);
            return pso;
        };
        gemv = pipeline("matmul_q4_0");
        mm = tile_pipeline(device, library, 0);
    }

    // Pipelines for a format other than Q4_0 (ternary not covered here).
    static uint32_t format_id(WeightFormat f) {
        switch (f) {
            case WeightFormat::Q4_K: return 2;
            case WeightFormat::Q6_K: return 3;
            case WeightFormat::Q4_1: return 4;
            case WeightFormat::Q5_0: return 5;
            case WeightFormat::Q8_0: return 6;
            default: return 0;
        }
    }
    void select(WeightFormat f) {
        if (f == WeightFormat::Q4_0) return;
        const uint32_t id = format_id(f);
        if (id == 0) fail("the standalone matmul takes Q4_0, Q4_1, Q5_0, Q8_0, Q4_K or Q6_K");
        NSError* error = nil;
        gemv = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"matmul_k"] error:&error];
        if (gemv == nil) fail("K-quant gemv pipeline");
        format = id;
        mm = tile_pipeline(device, library, id);
    }

    bool paired = false;
    void encode(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> w, id<MTLBuffer> in, id<MTLBuffer> half_in,
                id<MTLBuffer> out, std::size_t rows, std::size_t cols, std::size_t n, bool tiled) {
        if (tiled) {
            [enc setComputePipelineState:mm];
            [enc setBuffer:w offset:0 atIndex:0];
            [enc setBuffer:half_in offset:0 atIndex:1];
            [enc setBuffer:out offset:0 atIndex:2];
            const MatmulParams p{static_cast<uint32_t>(rows), static_cast<uint32_t>(cols), static_cast<uint32_t>(n), kStore, 0};
            [enc setBytes:&p length:sizeof(p) atIndex:3];
            [enc setBuffer:out offset:0 atIndex:4];
            [enc setBuffer:half_in offset:0 atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(rows / 64, (n + 63) / 64, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else {
            [enc setComputePipelineState:gemv];
            for (NSUInteger i = 0; i < 3; ++i) {
                [enc setBuffer:w offset:0 atIndex:i];
                [enc setBuffer:out offset:0 atIndex:4 + i];
            }
            [enc setBuffer:in offset:0 atIndex:3];
            // paired: gate and up are the same matrix, out = silu(w x) * (w x).
            const GemvParams p{{static_cast<uint32_t>(rows), 0, 0}, static_cast<uint32_t>(cols), static_cast<uint32_t>(n),
                               paired ? kGeluMul : kStore, paired ? 1u : 0u, {format, format, format}, 0};
            [enc setBytes:&p length:sizeof(p) atIndex:7];
            [enc setBuffer:in offset:0 atIndex:8];
            if (format == 0 && paired) {
                [enc dispatchThreadgroups:MTLSizeMake(((rows + 1) / 2 * n + 1) / 2, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            } else if (format == 0) {  // Q4_0: groups of 4 rows, 2 simdgroups per threadgroup
                [enc dispatchThreadgroups:MTLSizeMake(((rows + 3) / 4 * n + 1) / 2, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            } else if (paired) {  // K-quants: one item per row (of both matrices)
                [enc dispatchThreadgroups:MTLSizeMake((rows * n + 3) / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            } else {
                [enc dispatchThreadgroups:MTLSizeMake(((rows + 1) / 2 * n + 3) / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }
        }
    }
};

}  // namespace

std::vector<float> metal_matmul(WeightFormat format, std::span<const std::byte> weights, std::size_t rows,
                                std::size_t cols, std::span<const float> x, std::size_t n, bool tiled, bool paired) {
    @autoreleasepool {
        StandaloneMatmul m;
        m.select(format);
        m.paired = paired;
        id<MTLBuffer> w = [m.device newBufferWithBytes:weights.data() length:weights.size() options:MTLResourceStorageModeShared];
        id<MTLBuffer> in = [m.device newBufferWithBytes:x.data() length:x.size_bytes() options:MTLResourceStorageModeShared];
        id<MTLBuffer> half_in = [m.device newBufferWithLength:x.size() * 2 options:MTLResourceStorageModeShared];
        auto* halves = static_cast<std::uint16_t*>(half_in.contents);
        for (std::size_t i = 0; i < x.size(); ++i) halves[i] = float_to_half(x[i]);
        id<MTLBuffer> out = [m.device newBufferWithLength:rows * n * 4 options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb = [m.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        m.encode(enc, w, in, half_in, out, rows, cols, n, tiled);
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        std::vector<float> result(rows * n);
        std::memcpy(result.data(), out.contents, result.size() * sizeof(float));
        return result;
    }
}

double metal_matmul_q4_0_seconds(std::size_t rows, std::size_t cols, std::size_t n, bool tiled, int iterations) {
    return metal_matmul_seconds(WeightFormat::Q4_0, rows, cols, n, tiled, iterations);
}

double metal_matmul_seconds(WeightFormat format, std::size_t rows, std::size_t cols, std::size_t n, bool tiled,
                            int iterations) {
    @autoreleasepool {
        StandaloneMatmul m;
        m.select(format);
        id<MTLBuffer> w = [m.device newBufferWithLength:rows * cols / block_elements(format) * block_bytes(format)
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> in = [m.device newBufferWithLength:n * cols * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> half_in = [m.device newBufferWithLength:n * cols * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> out = [m.device newBufferWithLength:rows * n * 4 options:MTLResourceStorageModeShared];
        std::memset(w.contents, 0x37, w.length);
        std::memset(in.contents, 0, in.length);
        id<MTLCommandBuffer> cb = [m.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        for (int i = 0; i < iterations; ++i) m.encode(enc, w, in, half_in, out, rows, cols, n, tiled);
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        return (cb.GPUEndTime - cb.GPUStartTime) / iterations;
    }
}

double metal_gemv_tq2_0_seconds(std::size_t rows, std::size_t cols, int iterations, int variant) {
    @autoreleasepool {
        StandaloneMatmul m;
        NSError* error = nil;
        MTLFunctionConstantValues* constants = [MTLFunctionConstantValues new];
        const uint32_t v = static_cast<uint32_t>(variant);
        [constants setConstantValue:&v type:MTLDataTypeUInt atIndex:1];
        id<MTLFunction> fn = [m.library newFunctionWithName:@"matmul_tq2_0" constantValues:constants error:&error];
        if (fn == nil) fail(std::string("ternary gemv specialisation: ") + error.localizedDescription.UTF8String);
        id<MTLComputePipelineState> pso = [m.device newComputePipelineStateWithFunction:fn error:&error];
        if (pso == nil) fail("ternary gemv pipeline");
        id<MTLBuffer> w = [m.device newBufferWithLength:rows * cols / 256 * 66 options:MTLResourceStorageModeShared];
        id<MTLBuffer> in = [m.device newBufferWithLength:cols * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> out = [m.device newBufferWithLength:rows * 4 options:MTLResourceStorageModeShared];
        std::memset(w.contents, 0x5A, w.length);
        std::memset(in.contents, 0, in.length);
        id<MTLCommandBuffer> cb = [m.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        for (NSUInteger i = 0; i < 3; ++i) {
            [enc setBuffer:w offset:0 atIndex:i];
            [enc setBuffer:out offset:0 atIndex:4 + i];
        }
        [enc setBuffer:in offset:0 atIndex:3];
        [enc setBuffer:in offset:0 atIndex:8];
        const GemvParams p{{static_cast<uint32_t>(rows), 0, 0}, static_cast<uint32_t>(cols), 1, kStore, 0, {0, 0, 0}, 0};
        [enc setBytes:&p length:sizeof(p) atIndex:7];
        const std::size_t groups = (rows + kTernaryRows - 1) / kTernaryRows;
        for (int i = 0; i < iterations; ++i) {
            [enc dispatchThreadgroups:MTLSizeMake((groups + 1) / 2, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        return (cb.GPUEndTime - cb.GPUStartTime) / iterations;
    }
}

bool MetalSession::available() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return device != nil;
}

}  // namespace brisk
