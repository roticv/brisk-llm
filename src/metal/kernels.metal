// Metal kernels for the GPU forward pass. Compiled at runtime from this
// source (embedded in the binary), so no Xcode toolchain is needed.
//
// Conventions: activations are float32 rows of `dim` values, token-major;
// Q4_0 weights are the GGUF blocks (half scale, 16 bytes of nibbles) read in
// place from the memory-mapped file; the K/V cache is head-major:
// [kv head][position][head_dim].

#include <metal_stdlib>
using namespace metal;

struct Q4Block {
    half d;
    ushort qs[8];  // 16 bytes of nibbles: low nibbles elements 0-15, high 16-31
};

// What a matmul does with its result, so the elementwise work that follows
// a projection rides along in the epilogue instead of in its own pass.
enum : uint {
    kStore = 0,        // out = acc
    kResidual = 1,     // out = aux + acc (aux is the residual stream; out may alias it)
    kSiluMul = 2,      // out = silu(aux) * acc (aux is the gate projection)
    kReluSqrMul = 3,   // out = relu(aux)^2 * acc
};

struct MatmulParams {
    uint rows;
    uint cols;
    uint n;  // activation rows
    uint mode;
    uint write_half;  // also write the result as half, for the next tiled multiply
};

// Up to three matrices sharing the same input, multiplied in one dispatch so
// that small matrices (k, v) still fill the GPU.
struct GemvParams {
    uint rows[3];
    uint cols;
    uint n;
    uint mode;
};

struct NormParams {
    uint dim;
    float eps;
    uint half_out;  // write the half buffer instead of the float one
};

inline float epilogue(uint mode, float acc, float aux) {
    switch (mode) {
        case kResidual: return aux + acc;
        case kSiluMul: return aux / (1.0f + exp(-aux)) * acc;
        case kReluSqrMul: { const float r = max(aux, 0.0f); return r * r * acc; }
        default: return acc;
    }
}

struct RopeParams {
    uint n;             // tokens in the batch
    uint position0;     // position of the batch's first token
    uint head_count;
    uint kv_head_count;
    uint head_dim;
    uint max_context;
    float eps;
    float rope_base;
    uint has_qk_norm;   // Qwen3: RMS norm per head before RoPE
};

struct AttentionParams {
    uint n;
    uint position0;
    uint head_count;
    uint kv_head_count;
    uint head_dim;
    uint max_context;
    float scale;
    uint write_half;  // also write the output as half, for the tiled output projection
};

// ---- embedding ---------------------------------------------------------------

// One thread per (token, block of 32): dequantises the token's row.
kernel void embed_q4_0(device const uchar* table [[buffer(0)]],
                       device const int* tokens [[buffer(1)]],
                       device float* out [[buffer(2)]],
                       constant uint& dim [[buffer(3)]],
                       uint2 gid [[thread_position_in_grid]]) {
    const uint block = gid.x, t = gid.y;
    const uint blocks = dim / 32;
    device const Q4Block* b = (device const Q4Block*)(table + ((ulong)tokens[t] * blocks + block) * 18);
    const float d = float(b->d);
    device float* o = out + t * dim + block * 32;
    for (uint i = 0; i < 8; ++i) {
        const ushort w = b->qs[i];
        o[2 * i] = float(int(w & 0xF) - 8) * d;
        o[2 * i + 1] = float(int((w >> 8) & 0xF) - 8) * d;
        o[16 + 2 * i] = float(int((w >> 4) & 0xF) - 8) * d;
        o[16 + 2 * i + 1] = float(int(w >> 12) - 8) * d;
    }
}

// ---- RMS norm ----------------------------------------------------------------

// One threadgroup (256 threads) per row: out = x / sqrt(mean(x^2) + eps) * w.
kernel void rms_norm(device const float* x [[buffer(0)]],
                     device const float* w [[buffer(1)]],
                     device float* out [[buffer(2)]],
                     constant NormParams& p [[buffer(3)]],
                     device half* out_half [[buffer(4)]],
                     uint row [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[8];
    device const float* in = x + row * p.dim;
    float sum = 0.0f;
    for (uint i = tid; i < p.dim; i += 256) sum += in[i] * in[i];
    sum = simd_sum(sum);
    if (lane == 0) partial[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < 8; ++i) total += partial[i];
    const float scale = rsqrt(total / float(p.dim) + p.eps);
    if (p.half_out) {
        device half* o = out_half + row * p.dim;
        for (uint i = tid; i < p.dim; i += 256) o[i] = half(in[i] * scale * w[i]);
    } else {
        device float* o = out + row * p.dim;
        for (uint i = tid; i < p.dim; i += 256) o[i] = in[i] * scale * w[i];
    }
}

// ---- Q4_0 matrix multiply ---------------------------------------------------

// Blocks are 18 bytes, so the nibble bytes are only 2-byte aligned: a uint
// load there would be rounded down to a 4-byte boundary. Read ushort pairs.
inline uint q4_word(device const Q4Block* blk, uint k) {
    return uint(blk->qs[2 * k]) | (uint(blk->qs[2 * k + 1]) << 16);
}

// Dequantises a block's 32 weights as eight float4s (elements 0-15 then 16-31),
// with the nibbles unpacked by vector shifts.
inline void q4_block_dequant(device const Q4Block* blk, thread float4 (&v)[8]) {
    const float d = float(blk->d);
    for (uint k = 0; k < 4; ++k) {
        const uint w = q4_word(blk, k);
        v[k] = (float4(uint4(w) >> uint4(0, 8, 16, 24) & 0xF) - 8.0f) * d;
        v[4 + k] = (float4(uint4(w) >> uint4(4, 12, 20, 28) & 0xF) - 8.0f) * d;
    }
}

// One simdgroup per two rows of one activation row: the lanes take blocks in
// turn (both rows' blocks per lane, which keeps more loads in flight) and the
// partial sums are reduced across the simdgroup. The 4 simdgroups of a
// threadgroup take consecutive row pairs. Row pairs are numbered across the
// three matrices in turn.
kernel void matmul_q4_0(device const uchar* w0 [[buffer(0)]],
                        device const uchar* w1 [[buffer(1)]],
                        device const uchar* w2 [[buffer(2)]],
                        device const float* x [[buffer(3)]],
                        device float* out0 [[buffer(4)]],
                        device float* out1 [[buffer(5)]],
                        device float* out2 [[buffer(6)]],
                        constant GemvParams& p [[buffer(7)]],
                        device const float* aux [[buffer(8)]],
                        uint tg [[threadgroup_position_in_grid]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    const uint pairs0 = (p.rows[0] + 1) / 2, pairs1 = (p.rows[1] + 1) / 2, pairs2 = (p.rows[2] + 1) / 2;
    const uint item = tg * 4 + sg;
    if (item >= (pairs0 + pairs1 + pairs2) * p.n) return;
    uint pair = item / p.n;
    const uint t = item % p.n;
    device const uchar* w;
    device float* out;
    uint rows;
    if (pair < pairs0) {
        w = w0; out = out0; rows = p.rows[0];
    } else if (pair < pairs0 + pairs1) {
        pair -= pairs0; w = w1; out = out1; rows = p.rows[1];
    } else {
        pair -= pairs0 + pairs1; w = w2; out = out2; rows = p.rows[2];
    }
    const uint row0 = pair * 2;
    const bool two = row0 + 1 < rows;
    const uint blocks = p.cols / 32;
    device const uchar* wrow0 = w + (ulong)row0 * blocks * 18;
    device const uchar* wrow1 = wrow0 + (two ? blocks * 18 : 0);
    device const float4* xrow = (device const float4*)(x + t * p.cols);

    float acc0 = 0.0f, acc1 = 0.0f;
    for (uint b = lane; b < blocks; b += 32) {
        float4 v0[8], v1[8];
        q4_block_dequant((device const Q4Block*)(wrow0 + b * 18), v0);
        q4_block_dequant((device const Q4Block*)(wrow1 + b * 18), v1);
        float4 s0 = 0.0f, s1 = 0.0f;
        for (uint i = 0; i < 8; ++i) {
            const float4 xv = xrow[b * 8 + i];
            s0 += v0[i] * xv;
            s1 += v1[i] * xv;
        }
        acc0 += s0.x + s0.y + s0.z + s0.w;
        acc1 += s1.x + s1.y + s1.z + s1.w;
    }
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    if (lane == 0) {
        out[t * rows + row0] = epilogue(p.mode, acc0, aux[t * rows + row0]);
        if (two) out[t * rows + row0 + 1] = epilogue(p.mode, acc1, aux[t * rows + row0 + 1]);
    }
}

// Batched multiply on simdgroup matrices: each threadgroup (128 threads, 4
// simdgroups) produces a 64-token x 64-row tile of out[t][row], computed as
// X (tokens x k) times W^T (k x rows) so the accumulators store straight to
// the output. Per 32-column step every thread dequantises half a weight
// block and copies 16 activations into threadgroup memory, both laid out as
// contiguous 8x8 blocks (weights [k][row], activations [token][k]) so the
// matrix loads are contiguous; each simdgroup then accumulates a 32-token x
// 32-row quarter with 8x8 matrix multiplies. Each step's loads are issued
// before the previous step's multiplies so they overlap.
kernel void matmul_q4_0_mm(device const uchar* w [[buffer(0)]],
                           device const half* x [[buffer(1)]],
                           device float* out [[buffer(2)]],
                           constant MatmulParams& p [[buffer(3)]],
                           device const float* aux [[buffer(4)]],
                           device half* out_half [[buffer(5)]],
                           uint2 tg [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half4 shared4[2 * 64 * 32 / 4];  // declared as half4 for the alignment of the vector stores
    threadgroup half* sa = (threadgroup half*)shared4;  // weights: block (kb, rb) at 64 * (kb * 8 + rb), [k % 8][row % 8]
    threadgroup half* sb = sa + 64 * 32;                // activations: block (tb, kb) at 64 * (tb * 4 + kb), [t % 8][k % 8]
    const uint row0 = tg.x * 64, t0 = tg.y * 64;
    const uint blocks = p.cols / 32;

    simdgroup_float8x8 acc[4][4];  // [token block][row block]
    for (uint i = 0; i < 4; ++i) {
        for (uint j = 0; j < 4; ++j) acc[i][j] = simdgroup_float8x8(0.0f);
    }

    const uint st = (sg / 2) * 32, sr = (sg % 2) * 32;  // this simdgroup's quarter of the tile
    // This thread's weights (row r, the low or high nibbles of the step's
    // block, i.e. k = 0-15 or 16-31) and activation span (token t, columns
    // k0 .. k0 + 15) in each step.
    const uint r = tid / 2, high = tid % 2;
    const uint t = tid % 64, k0 = (tid / 64) * 16;
    const bool valid = t0 + t < p.n;
    device const uchar* wrow = w + (ulong)(row0 + r) * blocks * 18;
    device const half4* xsrc = (device const half4*)(x + (t0 + (valid ? t : 0)) * p.cols + k0);
    threadgroup half* wdst = sa + 64 * ((high * 2) * 8 + r / 8) + r % 8;  // k = high * 16, then + 8 per block
    threadgroup half4* xdst = (threadgroup half4*)(sb + 64 * ((t / 8) * 4 + k0 / 8) + 8 * (t % 8));
    const uint shift = high * 4;

    half wd;
    ushort wq[8];
    half4 xv[4];
    auto load = [&](uint b) {
        device const Q4Block* blk = (device const Q4Block*)(wrow + b * 18);
        wd = blk->d;
        for (uint i = 0; i < 8; ++i) wq[i] = blk->qs[i];
        for (uint k = 0; k < 4; ++k) xv[k] = valid ? xsrc[b * 8 + k] : half4(0.0h);
    };
    auto store = [&]() {
        for (uint k = 0; k < 4; ++k) {
            const uint wv = uint(wq[2 * k]) | (uint(wq[2 * k + 1]) << 16);
            const half4 v = (half4(uint4(wv) >> (uint4(0, 8, 16, 24) + shift) & 0xF) - 8.0h) * wd;
            // Elements 4k .. 4k+3 of this half: k index 4k + j; block 4k/8, row within block 4k%8 + j.
            const uint kb = (4 * k) / 8, kr = (4 * k) % 8;
            wdst[64 * 8 * kb + 8 * kr] = v.x;
            wdst[64 * 8 * kb + 8 * (kr + 1)] = v.y;
            wdst[64 * 8 * kb + 8 * (kr + 2)] = v.z;
            wdst[64 * 8 * kb + 8 * (kr + 3)] = v.w;
        }
        xdst[0] = xv[0];
        xdst[1] = xv[1];
        xdst[16] = xv[2];  // next k block: 64 halves = 16 half4 further
        xdst[17] = xv[3];
    };

    load(0);
    for (uint b = 0; b < blocks; ++b) {
        store();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (b + 1 < blocks) load(b + 1);

        for (uint kb = 0; kb < 4; ++kb) {
            simdgroup_half8x8 ma[4], mb[4];
            for (uint i = 0; i < 4; ++i) simdgroup_load(ma[i], sa + 64 * (kb * 8 + sr / 8 + i), 8);
            for (uint j = 0; j < 4; ++j) simdgroup_load(mb[j], sb + 64 * ((st / 8 + j) * 4 + kb), 8);
            for (uint j = 0; j < 4; ++j) {
                for (uint i = 0; i < 4; ++i) simdgroup_multiply_accumulate(acc[j][i], mb[j], ma[i], acc[j][i]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Stage through shared memory 32 tokens at a time, apply the epilogue and
    // write the tokens that exist.
    threadgroup float* tile = (threadgroup float*)shared4;  // [token][row], 32 x 64
    for (uint pass = 0; pass < 2; ++pass) {
        if (sg / 2 == pass) {
            for (uint j = 0; j < 4; ++j) {
                for (uint i = 0; i < 4; ++i) simdgroup_store(acc[j][i], tile + (8 * j) * 64 + sr + 8 * i, 64);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 32 * 64; i += 128) {
            const uint tt = pass * 32 + i / 64, rr = i % 64;
            if (t0 + tt < p.n) {
                const ulong o = (ulong)(t0 + tt) * p.rows + row0 + rr;
                const float v = epilogue(p.mode, tile[(i / 64) * 64 + rr], p.mode == kStore ? 0.0f : aux[o]);
                out[o] = v;
                if (p.write_half) out_half[o] = half(v);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ---- per-head norm, RoPE, cache write ----------------------------------------

// One simdgroup per (token, head). Heads 0 .. head_count-1 are query heads
// (normalised and rotated in place); the rest are key heads (normalised,
// rotated and written to the cache, with the matching value row copied).
kernel void qk_norm_rope(device float* q [[buffer(0)]],
                         device float* k_stage [[buffer(1)]],
                         device const float* v_stage [[buffer(2)]],
                         device const float* q_norm [[buffer(3)]],
                         device const float* k_norm [[buffer(4)]],
                         device float* k_cache [[buffer(5)]],
                         device float* v_cache [[buffer(6)]],
                         constant RopeParams& p [[buffer(7)]],
                         uint tg [[threadgroup_position_in_grid]],
                         uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    const uint item = tg * 4 + sg;
    const uint heads = p.head_count + p.kv_head_count;
    if (item >= p.n * heads) return;
    const uint t = item / heads, h = item % heads;
    const bool is_query = h < p.head_count;
    const uint kv = h - p.head_count;
    const uint position = p.position0 + t;
    device float* vec = is_query ? q + (t * p.head_count + h) * p.head_dim
                                 : k_stage + (t * p.kv_head_count + kv) * p.head_dim;
    device const float* norm = is_query ? q_norm : k_norm;

    if (p.has_qk_norm) {
        float sum = 0.0f;
        for (uint i = lane; i < p.head_dim; i += 32) sum += vec[i] * vec[i];
        sum = simd_sum(sum);
        const float scale = rsqrt(sum / float(p.head_dim) + p.eps);
        for (uint i = lane; i < p.head_dim; i += 32) vec[i] = vec[i] * scale * norm[i];
    }
    const uint half_dim = p.head_dim / 2;
    for (uint i = lane; i < half_dim; i += 32) {
        const float theta = float(position) * pow(p.rope_base, -2.0f * float(i) / float(p.head_dim));
        const float c = cos(theta), s = sin(theta);
        const float x0 = vec[i], x1 = vec[i + half_dim];
        vec[i] = x0 * c - x1 * s;
        vec[i + half_dim] = x0 * s + x1 * c;
    }
    if (!is_query) {
        device float* kc = k_cache + ((ulong)kv * p.max_context + position) * p.head_dim;
        device float* vc = v_cache + ((ulong)kv * p.max_context + position) * p.head_dim;
        device const float* vs = v_stage + (t * p.kv_head_count + kv) * p.head_dim;
        for (uint i = lane; i < p.head_dim; i += 32) {
            kc[i] = vec[i];
            vc[i] = vs[i];
        }
    }
}

// ---- attention --------------------------------------------------------------

// One threadgroup (128 threads) per (block of 8 tokens, key/value head): the
// block's queries for the heads sharing that key/value head are held in
// threadgroup memory, so each cached key and value row is read once for all
// of them. Scores go to a scratch region of 16 rows per threadgroup.
constant uint kTokenBlock = 8;
constant uint kMaxQueries = 16;  // kTokenBlock * (query heads per key/value head), at most 2 heads

kernel void attention(device const float* q [[buffer(0)]],
                      device const float* k_cache [[buffer(1)]],
                      device const float* v_cache [[buffer(2)]],
                      device float* out [[buffer(3)]],
                      device float* scores [[buffer(4)]],
                      constant AttentionParams& p [[buffer(5)]],
                      device half* out_half [[buffer(6)]],
                      uint tg [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float qs[kMaxQueries * 128];
    threadgroup float inv_sum[kMaxQueries];
    const uint kv_heads = p.kv_head_count;
    const uint group = p.head_count / kv_heads;  // query heads per key/value head
    const uint block = tg / kv_heads, kv = tg % kv_heads;
    const uint t0 = block * kTokenBlock;
    const uint tokens = min(kTokenBlock, p.n - t0);
    const uint queries = tokens * group;
    const uint visible_last = p.position0 + t0 + tokens;  // positions seen by the block's last token
    device const float* keys = k_cache + (ulong)kv * p.max_context * p.head_dim;
    device const float* values = v_cache + (ulong)kv * p.max_context * p.head_dim;
    device float* sc = scores + (ulong)tg * kMaxQueries * p.max_context;

    // Query i is token t0 + i / group, head kv * group + i % group; unused rows are zero.
    for (uint i = tid; i < kMaxQueries * 128; i += 128) {
        const uint qi = i / 128, d = i % 128;
        const uint t = t0 + qi / group, h = kv * group + qi % group;
        qs[i] = (qi < queries && d < p.head_dim) ? q[(t * p.head_count + h) * p.head_dim + d] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Scores as matrix blocks: Q (16 x 128) times the transpose of 8 key rows
    // at a time, on the simdgroup matrix unit. Rows past `queries` and
    // positions past `visible_last` (up to the next multiple of 8) are
    // computed too; they are never read.
    for (uint pos = sg * 8; pos < visible_last; pos += 32) {
        simdgroup_float8x8 acc0(0.0f), acc1(0.0f);
        for (uint d = 0; d < p.head_dim; d += 8) {
            simdgroup_float8x8 kb, q0, q1;
            simdgroup_load(kb, keys + pos * p.head_dim + d, p.head_dim, ulong2(0, 0), true);  // transposed: [d][pos]
            simdgroup_load(q0, qs + d, 128);
            simdgroup_load(q1, qs + 8 * 128 + d, 128);
            simdgroup_multiply_accumulate(acc0, q0, kb, acc0);
            simdgroup_multiply_accumulate(acc1, q1, kb, acc1);
        }
        simdgroup_store(acc0, sc + pos, p.max_context);
        simdgroup_store(acc1, sc + 8 * p.max_context + pos, p.max_context);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Softmax per query row over the positions its token sees.
    for (uint qi = sg; qi < queries; qi += 4) {
        const uint visible = p.position0 + t0 + qi / group + 1;
        device float* row = sc + qi * p.max_context;
        float m = -INFINITY;
        for (uint pos = lane; pos < visible; pos += 32) m = max(m, row[pos]);
        m = simd_max(m);
        float sum = 0.0f;
        for (uint pos = lane; pos < visible; pos += 32) {
            const float e = exp((row[pos] - m) * p.scale);
            row[pos] = e;
            sum += e;
        }
        sum = simd_sum(sum);
        if (lane == 0) inv_sum[qi] = 1.0f / sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Weighted values: one output dimension per thread, all queries at once,
    // each value row read once. Score chunks are staged in threadgroup memory
    // (zero past each token's visible positions) so the inner reads are broadcasts.
    threadgroup float stage[kMaxQueries * 64];
    float acc[kMaxQueries];
    for (uint qi = 0; qi < kMaxQueries; ++qi) acc[qi] = 0.0f;
    for (uint pos0 = 0; pos0 < visible_last; pos0 += 64) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < kMaxQueries * 64; i += 128) {
            const uint qi = i / 64, pos = pos0 + i % 64;
            const uint visible = p.position0 + t0 + qi / group + 1;
            stage[i] = (qi < queries && pos < visible) ? sc[qi * p.max_context + pos] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint end = min(64u, visible_last - pos0);
        if (tid < p.head_dim) {
            for (uint i = 0; i < end; ++i) {
                const float v = values[(pos0 + i) * p.head_dim + tid];
                for (uint qi = 0; qi < kMaxQueries; ++qi) acc[qi] += stage[qi * 64 + i] * v;
            }
        }
    }
    if (tid < p.head_dim) {
        for (uint qi = 0; qi < queries; ++qi) {
            const uint t = t0 + qi / group, h = kv * group + qi % group;
            const float v = acc[qi] * inv_sum[qi];
            out[(t * p.head_count + h) * p.head_dim + tid] = v;
            if (p.write_half) out_half[(t * p.head_count + h) * p.head_dim + tid] = half(v);
        }
    }
}
