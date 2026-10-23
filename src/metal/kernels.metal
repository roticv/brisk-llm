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
    kGeluMul = 4,      // out = gelu(aux) * acc (tanh approximation)
    kMulGelu = 5,      // out = gelu(acc) * aux (Gemma 4's per-layer embedding gate)
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
    uint paired;      // rows of w0 and w1 are paired (gate/up) and combined by the epilogue
    uint formats[3];  // matmul_k: each matrix's format (2 = Q4_K, 3 = Q6_K, 4 = Q4_1)
    uint interleave;  // unused (kept for layout)
};

struct NormParams {
    uint dim;
    float eps;
    uint half_out;  // write the half buffer instead of the float one
    uint residual;  // add the normalised row to the residual buffer instead (Gemma's post-norms)
    uint then_norm; // with residual: also write norm(residual) * w2 to out / out_half (the next pre-norm)
    float out_scale; // with residual: the updated residual is multiplied by this (Gemma 4's layer scalar)
    uint residual_half; // with residual: also write the updated residual as halves to residual_h (Gemma 4's gate input)
};

inline float epilogue(uint mode, float acc, float aux) {
    switch (mode) {
        case kResidual: return aux + acc;
        case kSiluMul: return aux / (1.0f + exp(-aux)) * acc;
        case kReluSqrMul: { const float r = max(aux, 0.0f); return r * r * acc; }
        case kGeluMul: {
            // The argument is clamped: tanh is saturated by then, and the fast-math tanh overflows past it.
            const float z = clamp(0.7978845608f * (aux + 0.044715f * aux * aux * aux), -20.0f, 20.0f);
            return 0.5f * aux * (1.0f + tanh(z)) * acc;
        }
        case kMulGelu: {
            const float z = clamp(0.7978845608f * (acc + 0.044715f * acc * acc * acc), -20.0f, 20.0f);
            return 0.5f * acc * (1.0f + tanh(z)) * aux;
        }
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
    float rope_scale;   // multiplies the position (Gemma's global layers)
    uint has_qk_norm;   // Qwen3: RMS norm per head before RoPE
    uint neox;          // pair element i with i + head_dim/2 (else 2i with 2i + 1)
    uint has_bias;      // Qwen2: biases added to the projections first
    uint has_factors;   // Llama 3: per-frequency divisors of the angles
    uint has_kv;        // 0: the layer shares another's cache; only the queries are processed (Gemma 4)
    uint value_norm;    // Gemma 4: V is RMS-normalised (no weight) before the cache copy
    uint rope_dim;      // dimensions rotated per head (head_dim, or fewer for Phi)
    float attn_factor;  // LongRoPE: the rotated values are scaled by this
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
    uint splits;      // position ranges per (token block, kv head); > 1 writes partials for attention_combine
    uint chunk;       // positions per split, a multiple of 8
    uint block_tokens;  // tokens per threadgroup of the batched kernel: kMaxQueries / group
    uint window;        // a token sees only the last `window` positions (0: all)
    uint head_blocks;   // threadgroups per (token or token block, kv head, split): the kv head's query heads are
                        // taken kMaxGroup (small kernel) or `heads` (batched) at a time
    uint heads;         // batched kernel: query heads per threadgroup (the group, capped at kMaxQueries)
};

// ---- embedding ---------------------------------------------------------------

// One thread per (token, block of 32): dequantises the token's row.
kernel void embed_q4_0(device const uchar* table [[buffer(0)]],
                       device const int* tokens [[buffer(1)]],
                       device float* out [[buffer(2)]],
                       constant uint& dim [[buffer(3)]],
                       constant float& scale [[buffer(4)]],
                       uint2 gid [[thread_position_in_grid]]) {
    const uint block = gid.x, t = gid.y;
    const uint blocks = dim / 32;
    device const Q4Block* b = (device const Q4Block*)(table + ((ulong)tokens[t] * blocks + block) * 18);
    const float d = float(b->d) * scale;
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
                     device float* residual [[buffer(5)]],
                     device const float* w2 [[buffer(6)]],
                     device half* residual_h [[buffer(7)]],
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
    if (p.residual) {
        device float* r = residual + row * p.dim;
        float sum2 = 0.0f;
        device half* rh = residual_h + row * p.dim;
        for (uint i = tid; i < p.dim; i += 256) {
            const float v = (r[i] + in[i] * scale * w[i]) * p.out_scale;
            r[i] = v;
            if (p.residual_half) rh[i] = half(v);
            sum2 += v * v;
        }
        if (!p.then_norm) return;
        // The next sub-block's pre-norm of the updated residual, in the same pass.
        sum2 = simd_sum(sum2);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) partial[sg] = sum2;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total2 = 0.0f;
        for (uint i = 0; i < 8; ++i) total2 += partial[i];
        const float scale2 = rsqrt(total2 / float(p.dim) + p.eps);
        if (p.half_out) {
            device half* o = out_half + row * p.dim;
            for (uint i = tid; i < p.dim; i += 256) o[i] = half(r[i] * scale2 * w2[i]);
        } else {
            device float* o = out + row * p.dim;
            for (uint i = tid; i < p.dim; i += 256) o[i] = r[i] * scale2 * w2[i];
        }
    } else if (p.half_out) {
        device half* o = out_half + row * p.dim;
        for (uint i = tid; i < p.dim; i += 256) o[i] = half(in[i] * scale * w[i]);
    } else {
        device float* o = out + row * p.dim;
        for (uint i = tid; i < p.dim; i += 256) o[i] = in[i] * scale * w[i];
    }
}

// Gemma's final logit softcap: x = cap * tanh(x / cap).
kernel void softcap(device float* x [[buffer(0)]],
                    constant uint& count [[buffer(1)]],
                    constant float& cap [[buffer(2)]],
                    uint i [[thread_position_in_grid]]) {
    if (i < count) x[i] = cap * precise::tanh(x[i] / cap);
}

kernel void to_half(device const float* in [[buffer(0)]],
                    device half* out [[buffer(1)]],
                    constant uint& count [[buffer(2)]],
                    uint i [[thread_position_in_grid]]) {
    if (i < count) out[i] = half(in[i]);
}

// LFM2's short convolution. in = [b | c | x] per token (3 * dim); the state
// bx = b * x of every position is kept in `cache`; out = c * sum_k w[d][k] *
// bx[pos + k + 1 - taps] (zero before the start). In-batch positions take bx
// from `in` directly, so the cache writes need no ordering.
struct ConvParams {
    uint n, dim, taps, position0, write_half;
};

kernel void shortconv(device const float* in [[buffer(0)]],
                      device const float* w [[buffer(1)]],
                      device float* cache [[buffer(2)]],
                      device float* out [[buffer(3)]],
                      device half* out_half [[buffer(4)]],
                      constant ConvParams& p [[buffer(5)]],
                      uint i [[thread_position_in_grid]]) {
    if (i >= p.n * p.dim) return;
    const uint t = i / p.dim, d = i % p.dim;
    const uint pos = p.position0 + t;
    auto bx_at = [&](uint q) -> float {  // the state at position q
        if (q >= p.position0) {
            const uint tq = q - p.position0;
            return in[tq * 3 * p.dim + d] * in[tq * 3 * p.dim + 2 * p.dim + d];
        }
        return cache[q * p.dim + d];
    };
    cache[pos * p.dim + d] = bx_at(pos);
    float sum = 0.0f;
    for (uint k = 0; k < p.taps; ++k) {
        if (pos + k + 1 < p.taps) continue;
        sum += w[d * p.taps + k] * bx_at(pos + k + 1 - p.taps);
    }
    const float v = in[t * 3 * p.dim + p.dim + d] * sum;
    out[i] = v;
    if (p.write_half) out_half[i] = half(v);
}

// Gemma 4's per-layer embedding inputs: ple[l][t][d] = (proj[t][l][d] + table[t][l][d] * table_scale) / sqrt(2),
// where proj has already been scaled and normalised per layer slice.
struct PleParams {
    uint n, layers, dim;
    float table_scale;
};

kernel void ple_combine(device const float* proj [[buffer(0)]],
                        device const float* table [[buffer(1)]],
                        device float* out [[buffer(2)]],
                        constant PleParams& p [[buffer(3)]],
                        uint i [[thread_position_in_grid]]) {
    const uint all = p.layers * p.dim;
    if (i >= p.n * all) return;
    const uint t = i / all, l = (i % all) / p.dim, d = i % p.dim;
    out[(l * p.n + t) * p.dim + d] = (proj[i] + table[i] * p.table_scale) * 0.70710678f;
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

// Q4_0 gemv structured after llama.cpp's mul_vec_q_n_f32: four row streams
// per simdgroup (rows r .. r + 3 of one matrix, or rows r, r + 1 of both the
// gate and up matrices when paired), two simdgroups per threadgroup, 16
// blocks in flight per simdgroup with two lanes per block, each lane taking
// half a block (8 bytes: elements il .. il + 7 and 16 + il .. 16 + il + 7).
// The nibbles are never shifted: the packed 16-bit words are masked and
// multiplied by activations pre-divided by 1, 256, 16 and 4096, and the -8
// offset is applied once per half block through the activations' sum. Rows
// past the end are clamped (read, not written).
constant uint kQ4Rows = 4;

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
    const bool paired = p.paired != 0;
    constexpr uint R = kQ4Rows, RM = kQ4Rows / 2;
    const uint groups0 = paired ? (p.rows[0] + RM - 1) / RM : (p.rows[0] + R - 1) / R;
    const uint groups1 = paired ? 0 : (p.rows[1] + R - 1) / R, groups2 = paired ? 0 : (p.rows[2] + R - 1) / R;
    const uint item = tg * 2 + sg;
    if (item >= (groups0 + groups1 + groups2) * p.n) return;
    uint group = item / p.n;
    const uint t = item % p.n;
    device const uchar* w;
    device float* out;
    uint rows;
    if (group < groups0) {
        w = w0; out = out0; rows = p.rows[0];
    } else if (group < groups0 + groups1) {
        group -= groups0; w = w1; out = out1; rows = p.rows[1];
    } else {
        group -= groups0 + groups1; w = w2; out = out2; rows = p.rows[2];
    }
    const uint blocks = p.cols / 32;
    const ulong row_bytes = (ulong)blocks * 18;
    const uint row0 = paired ? group * RM : group * R;
    device const uchar* stream[R];
    for (uint s = 0; s < R; ++s) {
        const uint r = paired ? row0 + s / 2 : row0 + s;
        device const uchar* base = (paired && (s % 2 == 1)) ? w1 : w;
        stream[s] = base + min(r, rows - 1) * row_bytes;
    }
    const uint ix = lane / 2, il = (lane % 2) * 8;  // block in flight, half of it
    device const float* yb = x + t * p.cols + ix * 32 + il;

    float sumf[R];
    for (uint s = 0; s < R; ++s) sumf[s] = 0.0f;
    for (uint ib = ix; ib < blocks; ib += 16) {
        float yl[16];
        float sumy = 0.0f;
        for (uint i = 0; i < 8; i += 2) {
            sumy += (yb[i] + yb[i + 1]) + (yb[i + 16] + yb[i + 17]);
            yl[i] = yb[i];
            yl[i + 1] = yb[i + 1] / 256.0f;
            yl[i + 8] = yb[i + 16] / 16.0f;
            yl[i + 9] = yb[i + 17] / 4096.0f;
        }
        for (uint s = 0; s < R; ++s) {
            device const Q4Block* blk = (device const Q4Block*)(stream[s] + ib * 18);
            device const ushort* qs = blk->qs + il / 2;
            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
            for (uint i = 0; i < 8; i += 2) {
                const uint q = qs[i / 2];
                acc0 += yl[i] * float(q & 0x000F);
                acc1 += yl[i + 1] * float(q & 0x0F00);
                acc2 += yl[i + 8] * float(q & 0x00F0);
                acc3 += yl[i + 9] * float(q & 0xF000);
            }
            sumf[s] += float(blk->d) * ((acc0 + acc1) + (acc2 + acc3) - 8.0f * sumy);
        }
        yb += 32 * 16;
    }
    float total[R];
    for (uint s = 0; s < R; ++s) total[s] = simd_sum(sumf[s]);
    if (lane == 0) {
        if (paired) {
            for (uint i = 0; i < RM; ++i) {
                if (row0 + i < rows) out[t * rows + row0 + i] = epilogue(p.mode, total[2 * i + 1], total[2 * i]);
            }
        } else {
            for (uint s = 0; s < R; ++s) {
                if (row0 + s < rows) out[t * rows + row0 + s] = epilogue(p.mode, total[s], aux[t * rows + row0 + s]);
            }
        }
    }
}

// ---- K-quants ----------------------------------------------------------------
//
// Q4_K: 256 values in 144 bytes: half d, half dmin, 12 bytes of 6-bit scales
// and minimums for 8 sub-blocks of 32, 128 bytes of nibbles (chunk c of 32
// bytes holds sub-block 2c in the low nibbles and 2c + 1 in the high).
// Q6_K: 256 values in 210 bytes: 128 bytes of low nibbles, 64 bytes of high
// bit pairs, 16 signed scales (one per 16 values), half d at byte 208.

// Scale and minimum of sub-block j of a Q4_K block.
inline void q4k_scale_min(device const uchar* scales, uint j, thread uint& scale, thread uint& min) {
    if (j < 4) {
        scale = scales[j] & 63;
        min = scales[j + 4] & 63;
    } else {
        scale = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        min = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    }
}

// Sub-block `sub` (32 values) of a Q4_K row, as eight float4s.
inline void q4k_subblock(device const uchar* row, uint sub, thread float4 (&v)[8]) {
    device const uchar* blk = row + (sub / 8) * 144;
    const uint j = sub % 8;
    const float d = float(*(device const half*)blk), dmin = float(*(device const half*)(blk + 2));
    uint scale, min;
    q4k_scale_min(blk + 4, j, scale, min);
    const float ds = d * float(scale), dm = dmin * float(min);
    device const uint* q = (device const uint*)(blk + 16 + (j / 2) * 32);  // 4-byte aligned: 144 k + 16 + 32 c
    const uint shift = (j % 2) * 4;
    for (uint k = 0; k < 8; ++k) {
        v[k] = float4(uint4(q[k]) >> (uint4(0, 8, 16, 24) + shift) & 0xF) * ds - dm;
    }
}

// Q5_K: Q4_K with 32 bytes of high bits between the scales and the nibbles
// (bit j of byte i is the fifth bit of value i of sub-block j); 176 bytes.
inline void q5k_subblock(device const uchar* row, uint sub, thread float4 (&v)[8]) {
    device const uchar* blk = row + (sub / 8) * 176;
    const uint j = sub % 8;
    const float d = float(*(device const half*)blk), dmin = float(*(device const half*)(blk + 2));
    uint scale, min;
    q4k_scale_min(blk + 4, j, scale, min);
    const float ds = d * float(scale), dm = dmin * float(min);
    device const uint* qh = (device const uint*)(blk + 16);                  // 176 k + 16: 4-byte aligned
    device const uint* q = (device const uint*)(blk + 48 + (j / 2) * 32);
    const uint shift = (j % 2) * 4;
    for (uint k = 0; k < 8; ++k) {
        const uint4 lo = uint4(q[k]) >> (uint4(0, 8, 16, 24) + shift) & 0xF;
        const uint4 hi = (uint4(qh[k]) >> (uint4(0, 8, 16, 24) + j) & 1) << 4;
        v[k] = float4(lo | hi) * ds - dm;
    }
}

// Sub-block `sub` (32 values) of a Q6_K row, as eight float4s.
inline void q6k_subblock(device const uchar* row, uint sub, thread float4 (&v)[8]) {
    device const uchar* blk = row + (sub / 8) * 210;
    const uint j = sub % 8, half_ = j / 4, m = j % 4;
    const float d = float(*(device const half*)(blk + 208));
    const float s0 = d * float(int(as_type<char>(blk[192 + 2 * j]))), s1 = d * float(int(as_type<char>(blk[192 + 2 * j + 1])));
    device const ushort* ql = (device const ushort*)(blk + half_ * 64 + (m % 2) * 32);  // 2-byte aligned
    device const ushort* qh = (device const ushort*)(blk + 128 + half_ * 32);
    const uint lshift = (m / 2) * 4, hshift = 2 * m;
    for (uint k = 0; k < 8; ++k) {
        const uint lo = uint(ql[2 * k]) | (uint(ql[2 * k + 1]) << 16);
        const uint hi = uint(qh[2 * k]) | (uint(qh[2 * k + 1]) << 16);
        const uint4 q = (uint4(lo) >> (uint4(0, 8, 16, 24) + lshift) & 0xF) | ((uint4(hi) >> (uint4(0, 8, 16, 24) + hshift) & 3) << 4);
        v[k] = (float4(q) - 32.0f) * (k < 4 ? s0 : s1);
    }
}

// Block `sub` (32 values) of a Q4_1 row (20-byte blocks: half d, half m, 16
// bytes of nibbles; value = d q + m), as eight float4s.
inline void q4_1_subblock(device const uchar* row, uint sub, thread float4 (&v)[8]) {
    device const uchar* blk = row + sub * 20;
    const float d = float(*(device const half*)blk), m = float(*(device const half*)(blk + 2));
    device const uint* q = (device const uint*)(blk + 4);  // 4-byte aligned: 20 k + 4
    for (uint k = 0; k < 4; ++k) {
        v[k] = float4(uint4(q[k]) >> uint4(0, 8, 16, 24) & 0xF) * d + m;
        v[4 + k] = float4(uint4(q[k]) >> uint4(4, 12, 20, 28) & 0xF) * d + m;
    }
}

// Block `sub` of a Q5_0 row (22-byte blocks: half d, 4 bytes holding bit 4 of
// each value, 16 nibble bytes; value = (q - 16) d), as eight float4s.
inline void q5_0_subblock(device const uchar* row, uint sub, thread float4 (&v)[8]) {
    device const uchar* blk = row + sub * 22;
    device const ushort* u = (device const ushort*)blk;  // 2-byte aligned
    const float d = float(as_type<half>(u[0]));
    const uint qh = uint(u[1]) | (uint(u[2]) << 16);
    for (uint k = 0; k < 4; ++k) {
        const uint w = uint(u[3 + 2 * k]) | (uint(u[4 + 2 * k]) << 16);  // nibble bytes 4k .. 4k+3
        const uint4 lo = uint4(w) >> uint4(0, 8, 16, 24) & 0xF, hi = uint4(w) >> uint4(4, 12, 20, 28) & 0xF;
        const uint4 blo = (uint4(qh) >> (uint4(0, 1, 2, 3) + 4 * k) & 1) << 4;        // bits 4k .. 4k+3
        const uint4 bhi = (uint4(qh) >> (uint4(16, 17, 18, 19) + 4 * k) & 1) << 4;    // bits 16+4k ..
        v[k] = (float4(lo | blo) - 16.0f) * d;
        v[4 + k] = (float4(hi | bhi) - 16.0f) * d;
    }
}

// Block `sub` of a Q8_0 row (34-byte blocks: half d, 32 signed bytes), as eight float4s.
inline void q8_0_subblock(device const uchar* row, uint sub, thread float4 (&v)[8]) {
    device const uchar* blk = row + sub * 34;
    device const ushort* u = (device const ushort*)blk;
    const float d = float(as_type<half>(u[0]));
    for (uint k = 0; k < 8; ++k) {
        const uint w = uint(u[1 + 2 * k]) | (uint(u[2 + 2 * k]) << 16);  // bytes 4k .. 4k+3
        const int4 q = (int4(uint4(w) >> uint4(0, 8, 16, 24) & 0xFF) ^ 0x80) - 0x80;  // sign-extend
        v[k] = float4(q) * d;
    }
}

// Row stride in bytes of a format with 32-value sub-blocks: 2 = Q4_K, 3 = Q6_K, 4 = Q4_1, 5 = Q5_0, 6 = Q8_0, 7 = Q5_K.
inline ulong subblock_row_bytes(uint format, uint cols) {
    switch (format) {
        case 2: return (ulong)(cols / 256) * 144;
        case 3: return (ulong)(cols / 256) * 210;
        case 7: return (ulong)(cols / 256) * 176;
        case 4: return (ulong)(cols / 32) * 20;
        case 5: return (ulong)(cols / 32) * 22;
        default: return (ulong)(cols / 32) * 34;
    }
}

inline void subblock_dequant(uint format, device const uchar* row, uint sub, thread float4 (&v)[8]) {
    switch (format) {
        case 2: q4k_subblock(row, sub, v); break;
        case 3: q6k_subblock(row, sub, v); break;
        case 7: q5k_subblock(row, sub, v); break;
        case 4: q4_1_subblock(row, sub, v); break;
        case 5: q5_0_subblock(row, sub, v); break;
        default: q8_0_subblock(row, sub, v); break;
    }
}

// The 16 values [16 high, 16 high + 16) of sub-block `sub` of a Q4_K / Q6_K /
// Q4_1 row as four half4s: what one thread of the tile stages per step.
inline void q4k_half16(device const uchar* row, uint sub, uint high, thread half4 (&v)[4]) {
    device const uchar* blk = row + (sub / 8) * 144;
    const uint j = sub % 8;
    uint scale, min;
    q4k_scale_min(blk + 4, j, scale, min);
    const half ds = half(float(*(device const half*)blk) * float(scale));
    const half dm = half(float(*(device const half*)(blk + 2)) * float(min));
    device const uint* q = (device const uint*)(blk + 16 + (j / 2) * 32) + high * 4;
    const uint shift = (j % 2) * 4;
    for (uint k = 0; k < 4; ++k) v[k] = half4(uint4(q[k]) >> (uint4(0, 8, 16, 24) + shift) & 0xF) * ds - dm;
}

inline void q6k_half16(device const uchar* row, uint sub, uint high, thread half4 (&v)[4]) {
    device const uchar* blk = row + (sub / 8) * 210;
    const uint j = sub % 8, half_ = j / 4, m = j % 4;
    const half scale = half(float(*(device const half*)(blk + 208)) * float(int(as_type<char>(blk[192 + 2 * j + high]))));
    device const ushort* ql = (device const ushort*)(blk + half_ * 64 + (m % 2) * 32) + high * 8;
    device const ushort* qh = (device const ushort*)(blk + 128 + half_ * 32) + high * 8;
    const uint lshift = (m / 2) * 4, hshift = 2 * m;
    for (uint k = 0; k < 4; ++k) {
        const uint lo = uint(ql[2 * k]) | (uint(ql[2 * k + 1]) << 16);
        const uint hi = uint(qh[2 * k]) | (uint(qh[2 * k + 1]) << 16);
        const uint4 q = (uint4(lo) >> (uint4(0, 8, 16, 24) + lshift) & 0xF) | ((uint4(hi) >> (uint4(0, 8, 16, 24) + hshift) & 3) << 4);
        v[k] = (half4(q) - 32.0h) * scale;
    }
}

inline void q41_half16(device const uchar* row, uint sub, uint high, thread half4 (&v)[4]) {
    device const uchar* blk = row + sub * 20;
    const half d = *(device const half*)blk, m = *(device const half*)(blk + 2);
    device const uint* q = (device const uint*)(blk + 4);
    const uint shift = high * 4;
    for (uint k = 0; k < 4; ++k) v[k] = half4(uint4(q[k]) >> (uint4(0, 8, 16, 24) + shift) & 0xF) * d + m;
}

inline void subblock_half16(uint format, device const uchar* row, uint sub, uint high, thread half4 (&v)[4]) {
    if (format == 2) q4k_half16(row, sub, high, v);
    else if (format == 3) q6k_half16(row, sub, high, v);
    else if (format == 4) q41_half16(row, sub, high, v);
    else {  // Q5_0 and Q8_0: through the float sub-block (the tile is not their hot path)
        float4 full[8];
        subblock_dequant(format, row, sub, full);
        for (uint k = 0; k < 4; ++k) v[k] = half4(full[high * 4 + k]);
    }
}

// Embedding rows of a K-quant table: one thread per (token, sub-block of 32).
constant uint kEmbedFormat [[function_constant(3)]];  // 2 = Q4_K, 3 = Q6_K, 4 = Q4_1, 5 = Q5_0, 6 = Q8_0

kernel void embed_k(device const uchar* table [[buffer(0)]],
                    device const int* tokens [[buffer(1)]],
                    device float* out [[buffer(2)]],
                    constant uint& dim [[buffer(3)]],
                    constant float& scale [[buffer(4)]],
                    uint2 gid [[thread_position_in_grid]]) {
    const uint sub = gid.x, t = gid.y;
    device const uchar* row = table + (ulong)tokens[t] * subblock_row_bytes(kEmbedFormat, dim);
    float4 v[8];
    subblock_dequant(kEmbedFormat, row, sub, v);
    device float4* o = (device float4*)(out + t * dim + sub * 32);
    for (uint k = 0; k < 8; ++k) o[k] = v[k] * scale;
}

// Gemv for K-quant matrices, each matrix's format given at run time (so a
// Q4_K q/k and a Q6_K v share one dispatch), with the dispatch shape of
// matmul_q4_0: one simdgroup per two rows and one activation row, 4
// simdgroups per threadgroup over consecutive row pairs, the lanes taking
// 32-value sub-blocks in turn.

kernel void matmul_k(device const uchar* w0 [[buffer(0)]],
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
    const bool paired = p.paired != 0;
    const uint pairs0 = paired ? p.rows[0] : (p.rows[0] + 1) / 2;
    const uint pairs1 = paired ? 0 : (p.rows[1] + 1) / 2, pairs2 = paired ? 0 : (p.rows[2] + 1) / 2;
    const uint item = tg * 4 + sg;
    if (item >= (pairs0 + pairs1 + pairs2) * p.n) return;
    uint pair = item / p.n;
    const uint t = item % p.n;
    device const uchar* w;
    device float* out;
    uint rows, format;
    if (pair < pairs0) {
        w = w0; out = out0; rows = p.rows[0]; format = p.formats[0];
    } else if (pair < pairs0 + pairs1) {
        pair -= pairs0; w = w1; out = out1; rows = p.rows[1]; format = p.formats[1];
    } else {
        pair -= pairs0 + pairs1; w = w2; out = out2; rows = p.rows[2]; format = p.formats[2];
    }
    const ulong row_bytes = subblock_row_bytes(format, p.cols);
    const uint row0 = paired ? pair : pair * 2;
    const bool two = paired || row0 + 1 < rows;
    device const uchar* wrow0 = w + row0 * row_bytes;
    device const uchar* wrow1 = paired ? w1 + row0 * row_bytes : wrow0 + (two ? row_bytes : 0);
    device const float4* xrow = (device const float4*)(x + t * p.cols);
    const uint subs = p.cols / 32;

    float acc0 = 0.0f, acc1 = 0.0f;
    if (format == 2) {
        // Q4_K after llama.cpp's mul_mv_q4_K: four blocks in flight per simdgroup,
        // eight lanes per block. Lane (iq, ir) of a block takes 8-value slices at
        // elements 64 iq + 8 ir (+32, +128, +160): the low and high nibbles of
        // 4 words of chunk iq and of chunk iq + 2, so the nibbles are never
        // shifted (activations are pre-divided by 256 and 16 instead), and the
        // sub-block scales/mins come from 16-bit words of the scale bytes.
        const uint ix = lane / 8, it = lane % 8, iq = it / 4, ir = it % 4;
        const uint blocks = p.cols / 256;
        device const float* y = x + t * p.cols + 64 * iq + 8 * ir;
        for (uint ib = ix; ib < blocks; ib += 4) {
            device const float* y4 = y + ib * 256;
            float yl[16], yh[16];
            float4 sumy = 0.0f;
            for (uint i = 0; i < 8; ++i) {
                yl[i] = y4[i];
                yl[i + 8] = y4[i + 32];
                yh[i] = y4[i + 128];
                yh[i + 8] = y4[i + 160];
                sumy += float4(yl[i], yl[i + 8], yh[i], yh[i + 8]);
            }
            for (uint r = 0; r < 2; ++r) {
                device const uchar* blk = (r == 0 ? wrow0 : wrow1) + ib * 144;
                device const ushort* sc = (device const ushort*)(blk + 4) + iq;
                device const ushort* q1 = (device const ushort*)(blk + 16) + 16 * iq + 4 * ir;
                device const ushort* q2 = q1 + 32;
                // Scales and minimums of sub-blocks 2iq, 2iq+1 (low nibbles) and 2iq+4, 2iq+5 (high):
                // bytes 0,1 scales of 2iq..2iq+1, 2,3 their mins, 4,5 scales of 2iq+4.., 6,7 their mins.
                ushort sc16[4];
                sc16[0] = sc[0] & 0x3f3f;
                sc16[1] = sc[2] & 0x3f3f;
                sc16[2] = ((sc[4] >> 0) & 0x0f0f) | ((sc[0] & 0xc0c0) >> 2);
                sc16[3] = ((sc[4] >> 4) & 0x0f0f) | ((sc[2] & 0xc0c0) >> 2);
                thread const uchar* sc8 = (thread const uchar*)sc16;
                float4 lo = 0.0f, hi = 0.0f;
                for (uint i = 0; i < 4; ++i) {
                    const uint w1 = q1[i], w2 = q2[i];
                    lo[0] += yl[2 * i] * float(w1 & 0x000F);
                    lo[1] += yl[2 * i + 1] * float(w1 & 0x0F00);
                    lo[2] += yl[2 * i + 8] * float(w1 & 0x00F0);
                    lo[3] += yl[2 * i + 9] * float(w1 & 0xF000);
                    hi[0] += yh[2 * i] * float(w2 & 0x000F);
                    hi[1] += yh[2 * i + 1] * float(w2 & 0x0F00);
                    hi[2] += yh[2 * i + 8] * float(w2 & 0x00F0);
                    hi[3] += yh[2 * i + 9] * float(w2 & 0xF000);
                }
                const float d = float(*(device const half*)blk), dmin = float(*(device const half*)(blk + 2));
                const float v = d * ((lo[0] + lo[1] * (1.0f / 256.0f)) * float(sc8[0]) +
                                     (lo[2] + lo[3] * (1.0f / 256.0f)) * float(sc8[1]) * (1.0f / 16.0f) +
                                     (hi[0] + hi[1] * (1.0f / 256.0f)) * float(sc8[4]) +
                                     (hi[2] + hi[3] * (1.0f / 256.0f)) * float(sc8[5]) * (1.0f / 16.0f)) -
                                dmin * (sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3]) + sumy[2] * float(sc8[6]) + sumy[3] * float(sc8[7]));
                if (r == 0) acc0 += v;
                else acc1 += v;
            }
        }
    }
    for (uint sub = lane; sub < (format == 2 ? 0u : subs); sub += 32) {  // the other formats: a sub-block per lane
        float4 va[8], vb[8];
        subblock_dequant(format, wrow0, sub, va);
        subblock_dequant(format, wrow1, sub, vb);
        float4 s0 = 0.0f, s1 = 0.0f;
        for (uint i = 0; i < 8; ++i) {
            const float4 xv = xrow[sub * 8 + i];
            s0 += va[i] * xv;
            s1 += vb[i] * xv;
        }
        acc0 += s0.x + s0.y + s0.z + s0.w;
        acc1 += s1.x + s1.y + s1.z + s1.w;
    }
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    if (lane == 0) {
        if (paired) {
            out[t * rows + row0] = epilogue(p.mode, acc1, acc0);
        } else {
            out[t * rows + row0] = epilogue(p.mode, acc0, aux[t * rows + row0]);
            if (two) out[t * rows + row0 + 1] = epilogue(p.mode, acc1, aux[t * rows + row0 + 1]);
        }
    }
}

// ---- TQ2_0 (ternary) ---------------------------------------------------------
//
// A block is 256 values in 64 bytes of 2-bit codes (0, 1, 2 for -1, 0, +1)
// followed by a half scale: 66 bytes. Byte j + m (j in {0, 32}, m < 32)
// holds, at bits 2l, element j * 4 + 32 * l + m. So 32 consecutive elements
// starting at a multiple of 32 are one bit pair of 32 consecutive bytes.

// Four consecutive ternary codes at bit pair l of a word's bytes, as float4
// in {0, 1, 2}; `bytes` is the word spread over the components (word >> 0,
// 8, 16, 24). The -1 offset is applied by the caller as a subtraction of the
// activations' sum: sum((q - 1) x) = sum(q x) - sum(x).
inline float4 tq2_codes(uint4 bytes, uint l) {
    return float4(extract_bits(bytes, 2 * l, 2));
}

// Gemv for TQ2_0 matrices, structured after llama.cpp's kernel_mul_mv_tq2_0:
// four row streams per simdgroup (rows r .. r + 3 of one matrix, or rows r,
// r + 1 of both the gate and up matrices when paired), two simdgroups per
// threadgroup, and within a simdgroup four blocks in flight with eight lanes
// per block, each lane taking 4 bytes of each 32-byte half of its block
// (elements 4 htg + 32 l + 128 h, l = 0..3). The decode is float: with v a
// byte as float and f_l = floor(v / 4^l) = q_l + 4 f_{l+1},
// sum_l q_l y_l = sum_l f_l (y_l - 4 y_{l-1}); the coefficients and the
// activation sum (for the -1 offset) are shared by the four rows. Rows past
// the end are clamped (read, not written).
constant uint kTernaryRows = 8;  // row streams per simdgroup (the host dispatches to match)
// Experiment switch for the standalone benchmark (unset in production): 1
// skips the decode and sums the raw bytes, to measure the access pattern's ceiling.
constant uint kTernaryVariant [[function_constant(1)]];
constant bool kTernaryHasVariant = is_function_constant_defined(kTernaryVariant);

kernel void matmul_tq2_0(device const uchar* w0 [[buffer(0)]],
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
    const bool paired = p.paired != 0;
    constexpr uint R = kTernaryRows;           // row streams per simdgroup
    constexpr uint RM = kTernaryRows / 2;      // rows per matrix when paired
    const uint groups0 = paired ? (p.rows[0] + RM - 1) / RM : (p.rows[0] + R - 1) / R;
    const uint groups1 = paired ? 0 : (p.rows[1] + R - 1) / R, groups2 = paired ? 0 : (p.rows[2] + R - 1) / R;
    const uint item = tg * 2 + sg;
    if (item >= (groups0 + groups1 + groups2) * p.n) return;
    uint group = item / p.n;
    const uint t = item % p.n;
    device const uchar* w;
    device float* out;
    uint rows;
    if (group < groups0) {
        w = w0; out = out0; rows = p.rows[0];
    } else if (group < groups0 + groups1) {
        group -= groups0; w = w1; out = out1; rows = p.rows[1];
    } else {
        group -= groups0 + groups1; w = w2; out = out2; rows = p.rows[2];
    }
    const uint blocks = p.cols / 256;
    const ulong row_bytes = (ulong)blocks * 66;
    const uint row0 = paired ? group * RM : group * R;
    device const uchar* stream[R];
    for (uint s = 0; s < R; ++s) {
        const uint r = paired ? row0 + s / 2 : row0 + s;
        device const uchar* base = (paired && (s % 2 == 1)) ? w1 : w;
        stream[s] = base + min(r, rows - 1) * row_bytes;
    }
    const uint blk = lane / 8, htg = lane % 8;  // block in flight, lane within it
    device const float4* yb4 = (device const float4*)(x + t * p.cols + 4 * htg + blk * 256);

    float sumf[R];
    for (uint s = 0; s < R; ++s) sumf[s] = 0.0f;
    for (uint ib = blk; ib < blocks; ib += 4) {
        for (uint h = 0; h < 2; ++h) {
            // y_l holds the activations of the lane's 4 bytes at bit pair l; the
            // coefficients are vectors over the 4 bytes, so the decode is 4-wide.
            const float4 y0 = yb4[0 + 32 * h], y1 = yb4[8 + 32 * h], y2 = yb4[16 + 32 * h], y3 = yb4[24 + 32 * h];
            const float4 c1 = y1 - 4.0f * y0, c2 = y2 - 4.0f * y1, c3 = y3 - 4.0f * y2;
            const float4 ysum = (y0 + y1) + (y2 + y3);
            const float sumy = (ysum.x + ysum.y) + (ysum.z + ysum.w);
            // The decode runs in half precision (bytes and their floor divisions
            // are exact in half; the activations lose their low bits, which the
            // perplexity check does not notice), which this GPU issues faster.
            const uint variant = kTernaryHasVariant ? kTernaryVariant : 0;
            // Scaled by 1/64 so that byte x coefficient products stay well inside half's range.
            const half4 hy0 = half4(y0 * (1.0f / 64)), hc1 = half4(c1 * (1.0f / 64)), hc2 = half4(c2 * (1.0f / 64)),
                        hc3 = half4(c3 * (1.0f / 64));
            for (uint s = 0; s < R; ++s) {
                device const uchar* b = stream[s] + ib * 66;
                const half4 v = half4(*(device const packed_uchar4*)(b + 4 * htg + 32 * h));
                if (variant == 1) {  // loads only
                    sumf[s] += float((v.x + v.y) + (v.z + v.w)) + float(*(device const half*)(b + 64));
                    continue;
                }
                half4 r = v * hy0;
                r = fma(floor(v * 0.25h), hc1, r);
                r = fma(floor(v * 0.0625h), hc2, r);
                r = fma(floor(v * 0.015625h), hc3, r);
                const float sum = float((r.x + r.y) + (r.z + r.w)) * 64.0f - sumy;
                sumf[s] += float(*(device const half*)(b + 64)) * sum;
            }
        }
        yb4 += 256;  // 4 blocks on
    }
    float total[R];
    for (uint s = 0; s < R; ++s) total[s] = simd_sum(sumf[s]);
    if (lane == 0) {
        if (paired) {
            for (uint i = 0; i < RM; ++i) {
                if (row0 + i < rows) out[t * rows + row0 + i] = epilogue(p.mode, total[2 * i + 1], total[2 * i]);
            }
        } else {
            for (uint s = 0; s < R; ++s) {
                if (row0 + s < rows) out[t * rows + row0 + s] = epilogue(p.mode, total[s], aux[t * rows + row0 + s]);
            }
        }
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
// The tile's weight format: 0 = Q4_0, 1 = TQ2_0, 2 = Q4_K, 3 = Q6_K, 4 = Q4_1, 5 = Q5_0, 6 = Q8_0 (a function constant, so each is its own pipeline).
constant uint kTileFormat [[function_constant(0)]];

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
    const uint blocks = p.cols / 32;  // steps of 32 columns
    const bool ternary = kTileFormat == 1;
    const bool kquant = kTileFormat >= 2;

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
    const ulong row_bytes = kTileFormat == 0 ? (ulong)blocks * 18
                            : kTileFormat == 1 ? (ulong)(p.cols / 256) * 66 : subblock_row_bytes(kTileFormat, p.cols);
    device const uchar* wrow = w + (ulong)(row0 + r) * row_bytes;
    device const half4* xsrc = (device const half4*)(x + (t0 + (valid ? t : 0)) * p.cols + k0);
    threadgroup half* wdst = sa + 64 * ((high * 2) * 8 + r / 8) + r % 8;  // k = high * 16, then + 8 per block
    threadgroup half4* xdst = (threadgroup half4*)(sb + 64 * ((t / 8) * 4 + k0 / 8) + 8 * (t % 8));
    const uint shift = high * 4;

    half wd;
    ushort wq[8];
    half4 xv[4];
    uint tl = 0;  // ternary: the bit pair of this step's 32 columns
    half4 kv[4];  // K-quants: this thread's 16 values of the step's sub-block, dequantised at load
    auto load = [&](uint b) {
        if (kquant) {
            subblock_half16(kTileFormat, wrow, b, high, kv);
        } else if (ternary) {
            // Columns [32 b, + 32) are bit pair l of bytes j .. j + 31 of block 32 b / 256;
            // this thread's 16 columns are bytes j + 16 high .. + 15: 4 words.
            const uint c0 = b * 32;
            device const uchar* blk = wrow + (c0 / 256) * 66;
            const uint j = ((c0 % 256) / 128) * 32;
            tl = (c0 % 128) / 32;
            wd = *(device const half*)(blk + 64);
            device const ushort* u = (device const ushort*)(blk + j + 16 * high);
            for (uint i = 0; i < 8; ++i) wq[i] = u[i];
        } else {
            device const Q4Block* blk = (device const Q4Block*)(wrow + b * 18);
            wd = blk->d;
            for (uint i = 0; i < 8; ++i) wq[i] = blk->qs[i];
        }
        for (uint k = 0; k < 4; ++k) xv[k] = valid ? xsrc[b * 8 + k] : half4(0.0h);
    };
    auto store = [&]() {
        for (uint k = 0; k < 4; ++k) {
            const uint wv = uint(wq[2 * k]) | (uint(wq[2 * k + 1]) << 16);
            const half4 v = kquant ? kv[k]
                            : ternary ? (half4(extract_bits(uint4(wv) >> uint4(0, 8, 16, 24), 2 * tl, 2)) - 1.0h) * wd
                                      : (half4(uint4(wv) >> (uint4(0, 8, 16, 24) + shift) & 0xF) - 8.0h) * wd;
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
                         device const float* q_bias [[buffer(8)]],
                         device const float* k_bias [[buffer(9)]],
                         device const float* v_bias [[buffer(10)]],
                         device const float* factors [[buffer(11)]],
                         device float* v_stage_out [[buffer(12)]],
                         uint tg [[threadgroup_position_in_grid]],
                         uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    const uint item = tg * 4 + sg;
    const uint heads = p.head_count + (p.has_kv ? p.kv_head_count : 0);
    if (item >= p.n * heads) return;
    const uint t = item / heads, h = item % heads;
    const bool is_query = h < p.head_count;
    const uint kv = h - p.head_count;
    const uint position = p.position0 + t;
    device float* vec = is_query ? q + (t * p.head_count + h) * p.head_dim
                                 : k_stage + (t * p.kv_head_count + kv) * p.head_dim;
    device const float* norm = is_query ? q_norm : k_norm;

    if (p.has_bias) {
        device const float* bias = is_query ? q_bias + h * p.head_dim : k_bias + kv * p.head_dim;
        for (uint i = lane; i < p.head_dim; i += 32) vec[i] += bias[i];
        if (!is_query) {  // the value head's bias, added in place before the cache copy below
            device float* vs = v_stage_out + (t * p.kv_head_count + kv) * p.head_dim;
            for (uint i = lane; i < p.head_dim; i += 32) vs[i] += v_bias[kv * p.head_dim + i];
        }
    }
    if (p.has_qk_norm) {
        float sum = 0.0f;
        for (uint i = lane; i < p.head_dim; i += 32) sum += vec[i] * vec[i];
        sum = simd_sum(sum);
        const float scale = rsqrt(sum / float(p.head_dim) + p.eps);
        for (uint i = lane; i < p.head_dim; i += 32) vec[i] = vec[i] * scale * norm[i];
    }
    // Only the first rope_dim dimensions rotate (all of them unless the model
    // rotates part of the head, as Phi does); the rotated values take attn_factor.
    const uint half_dim = p.rope_dim / 2;
    for (uint i = lane; i < half_dim; i += 32) {
        float theta = float(position) * p.rope_scale * pow(p.rope_base, -2.0f * float(i) / float(p.rope_dim));
        if (p.has_factors) theta /= factors[i];
        const float c = cos(theta) * p.attn_factor, s = sin(theta) * p.attn_factor;
        const uint a = p.neox ? i : 2 * i, b = p.neox ? i + half_dim : 2 * i + 1;
        const float x0 = vec[a], x1 = vec[b];
        vec[a] = x0 * c - x1 * s;
        vec[b] = x0 * s + x1 * c;
    }
    if (!is_query) {
        device float* kc = k_cache + ((ulong)kv * p.max_context + position) * p.head_dim;
        device float* vc = v_cache + ((ulong)kv * p.max_context + position) * p.head_dim;
        device const float* vs = v_stage + (t * p.kv_head_count + kv) * p.head_dim;
        float vscale = 1.0f;
        if (p.value_norm) {
            float sum = 0.0f;
            for (uint i = lane; i < p.head_dim; i += 32) sum += vs[i] * vs[i];
            sum = simd_sum(sum);
            vscale = rsqrt(sum / float(p.head_dim) + p.eps);
        }
        for (uint i = lane; i < p.head_dim; i += 32) {
            kc[i] = vec[i];
            vc[i] = vs[i] * vscale;
        }
    }
}

// ---- attention --------------------------------------------------------------

// One threadgroup (128 threads) per (block of 8 tokens, key/value head): the
// block's queries for the heads sharing that key/value head are held in
// threadgroup memory, so each cached key and value row is read once for all
// of them. Scores go to a scratch region of 16 rows per threadgroup.
// Head-dimension capacity of the attention kernels: 128, 256 or 512 (a
// function constant, so models with small heads keep the smaller
// threadgroup memory). The batched kernel holds 16 query rows (tokens per
// block * query heads per key/value head, p.block_tokens * group) up to 256
// wide, 8 at 512: 16 KB of queries either way.
constant uint kAttnDim [[function_constant(5)]];
constant uint kMaxQueries = kAttnDim == 512 ? 8 : 16;

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
    threadgroup float qs[16 * 256];  // kMaxQueries rows of kAttnDim
    threadgroup float inv_sum[16];
    threadgroup float row_max[16];
    const uint kv_heads = p.kv_head_count;
    const uint all_heads = p.head_count / kv_heads;  // query heads per key/value head
    // Threadgroup: (token block, kv head, head block, split); it takes `group`
    // = p.heads of the kv head's query heads (all of them when they fit).
    const uint split = tg % p.splits;
    const uint hb = (tg / p.splits) % p.head_blocks;
    const uint item = (tg / p.splits) / p.head_blocks;
    const uint block = item / kv_heads, kv = item % kv_heads;
    const uint head0 = kv * all_heads + hb * p.heads;
    const uint group = min(p.heads, all_heads - hb * p.heads);
    const uint t0 = block * p.block_tokens;
    const uint tokens = min(p.block_tokens, p.n - t0);
    const uint queries = tokens * group;
    // This threadgroup's positions: [pos0, visible_last), a slice of what the block's last token sees.
    const uint pos0 = split * p.chunk;
    const uint visible_last = min(p.position0 + t0 + tokens, pos0 + p.chunk);
    device const float* keys = k_cache + (ulong)kv * p.max_context * p.head_dim;
    device const float* values = v_cache + (ulong)kv * p.max_context * p.head_dim;
    // Score rows are indexed from pos0 so a split uses only `chunk` columns.
    const uint row_stride = p.splits > 1 ? p.chunk : p.max_context;
    device float* sc = scores + (ulong)tg * kMaxQueries * row_stride - pos0;

    // Query i is token t0 + i / group, head kv * group + i % group; unused rows are zero.
    for (uint i = tid; i < kMaxQueries * kAttnDim; i += 128) {
        const uint qi = i / kAttnDim, d = i % kAttnDim;
        const uint t = t0 + qi / group, h = head0 + qi % group;
        qs[i] = (qi < queries && d < p.head_dim) ? q[(t * p.head_count + h) * p.head_dim + d] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Scores as matrix blocks: Q (16 x 128) times the transpose of 8 key rows
    // at a time, on the simdgroup matrix unit. Rows past `queries` and
    // positions past `visible_last` (up to the next multiple of 8) are
    // computed too; they are never read.
    for (uint pos = pos0 + sg * 8; pos < visible_last; pos += 32) {
        simdgroup_float8x8 acc0(0.0f), acc1(0.0f);
        for (uint d = 0; d < p.head_dim; d += 8) {
            simdgroup_float8x8 kb, q0, q1;
            simdgroup_load(kb, keys + pos * p.head_dim + d, p.head_dim, ulong2(0, 0), true);  // transposed: [d][pos]
            simdgroup_load(q0, qs + d, kAttnDim);
            simdgroup_multiply_accumulate(acc0, q0, kb, acc0);
            if (kMaxQueries == 16) {
                simdgroup_load(q1, qs + 8 * kAttnDim + d, kAttnDim);
                simdgroup_multiply_accumulate(acc1, q1, kb, acc1);
            }
        }
        simdgroup_store(acc0, sc + pos, row_stride);
        if (kMaxQueries == 16) simdgroup_store(acc1, sc + 8 * row_stride + pos, row_stride);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Softmax per query row over the positions its token sees; the rest of
    // the row up to the padded end is zeroed (rows past `queries` too) so the
    // value pass can multiply whole 8-position blocks.
    const uint padded = (visible_last + 7) / 8 * 8;
    for (uint qi = sg; qi < kMaxQueries; qi += 4) {
        const uint own = p.position0 + t0 + qi / group + 1;  // positions the query's token has seen
        const uint visible = qi < queries ? min(own, visible_last) : 0;
        const uint start = (p.window != 0 && own > p.window) ? own - p.window : 0;  // sliding window
        device float* row = sc + qi * row_stride;
        float m = -INFINITY;
        for (uint pos = pos0 + lane; pos < visible; pos += 32) {
            if (pos >= start) m = max(m, row[pos]);
        }
        m = simd_max(m) * p.scale;  // the row's maximum in scaled units (-inf when it sees nothing here)
        float sum = 0.0f;
        for (uint pos = pos0 + lane; pos < padded; pos += 32) {
            const float e = (pos < visible && pos >= start) ? exp(row[pos] * p.scale - m) : 0.0f;
            row[pos] = e;
            sum += e;
        }
        sum = simd_sum(sum);
        if (lane == 0) {
            inv_sum[qi] = qi < queries && sum > 0.0f ? 1.0f / sum : 0.0f;
            row_max[qi] = m;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Weighted values on the matrix unit: P (16 queries x 8 positions) times
    // V (8 positions x 32 dimensions) per simdgroup, each simdgroup taking a
    // 32-dimension slice of a 128-dimension half (two halves for 256-wide
    // heads); rows past `queries` and positions past `visible_last` carry
    // zero weights.
    threadgroup float stage[16 * 128];  // kMaxQueries rows
    for (uint dbase = 0; dbase < p.head_dim; dbase += 128) {
        const uint d0 = dbase + sg * 32;
        simdgroup_float8x8 acc[2][4];
        for (uint i = 0; i < 2; ++i) {
            for (uint j = 0; j < 4; ++j) acc[i][j] = simdgroup_float8x8(0.0f);
        }
        if (d0 < p.head_dim) {
            for (uint pos = pos0; pos < visible_last; pos += 8) {
                simdgroup_float8x8 pb[2], vb[4];
                simdgroup_load(pb[0], sc + pos, row_stride);
                if (kMaxQueries == 16) simdgroup_load(pb[1], sc + 8 * row_stride + pos, row_stride);
                for (uint j = 0; j < 4; ++j) simdgroup_load(vb[j], values + pos * p.head_dim + d0 + j * 8, p.head_dim);
                for (uint i = 0; i < (kMaxQueries == 16 ? 2u : 1u); ++i) {
                    for (uint j = 0; j < 4; ++j) simdgroup_multiply_accumulate(acc[i][j], pb[i], vb[j], acc[i][j]);
                }
            }
        }
        // Stage [query][dim slice] and write the queries that exist, normalised.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < 2; ++i) {
            for (uint j = 0; j < 4; ++j) simdgroup_store(acc[i][j], stage + (8 * i) * 128 + (sg * 32) + j * 8, 128);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint dims = min(128u, p.head_dim - dbase);
        if (p.splits > 1) {
            // Partial result for attention_combine: unnormalised sums with the
            // row's maximum and denominator, at [token][head][split].
            for (uint i = tid; i < queries * dims; i += 128) {
                const uint qi = i / dims, d = dbase + i % dims;
                const uint t = t0 + qi / group, h = head0 + qi % group;
                const ulong slot = ((ulong)t * p.head_count + h) * p.splits + split;
                out[slot * (p.head_dim + 2) + d] = stage[qi * 128 + d - dbase];
                if (d == 0) {
                    out[slot * (p.head_dim + 2) + p.head_dim] = row_max[qi];
                    out[slot * (p.head_dim + 2) + p.head_dim + 1] = inv_sum[qi] > 0.0f ? 1.0f / inv_sum[qi] : 0.0f;
                }
            }
            continue;
        }
        for (uint i = tid; i < queries * dims; i += 128) {
            const uint qi = i / dims, d = dbase + i % dims;
            const uint t = t0 + qi / group, h = head0 + qi % group;
            const float v = stage[qi * 128 + d - dbase] * inv_sum[qi];
            out[(t * p.head_count + h) * p.head_dim + d] = v;
            if (p.write_half) out_half[(t * p.head_count + h) * p.head_dim + d] = half(v);
        }
    }
}

// Attention for a few tokens: one threadgroup (4 simdgroups) per (token,
// key/value head, split). Each simdgroup walks a quarter of the split's
// positions; the lanes hold 4 dimensions each, so a key or value row is one
// coalesced 512-byte read serving every query head of the group. Writes the
// same partials as the batched kernel for attention_combine.
constant uint kSmallChunk = 128;  // positions per split (the host dispatches ceil(visible / kSmallChunk) splits)
// Query heads per key/value head the small kernel handles: 8 with 128-wide
// heads, 4 with 256-wide ones (the per-simdgroup sums must fit threadgroup memory).
constant uint kD4 = kAttnDim / 128;  // float4s per lane
constant uint kMaxGroup = 8 / kD4;   // query heads per threadgroup: 8, 4 or 2

kernel void attention_small(device const float* q [[buffer(0)]],
                            device const float* k_cache [[buffer(1)]],
                            device const float* v_cache [[buffer(2)]],
                            device float* partial [[buffer(3)]],
                            constant AttentionParams& p [[buffer(4)]],
                            uint tg [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sc[8 * kSmallChunk];
    // Per-simdgroup value sums: slot g * kD4 + j holds query g's float4 j of each lane (16 KB either way).
    threadgroup float4 part[4][8][32];
    threadgroup float stats[8][2];  // max, sum per query
    // Threadgroup: (token, kv head, head block, split); the head block takes
    // query heads hb * kMaxGroup .. of the kv head's group.
    const uint all_heads = p.head_count / p.kv_head_count;
    const uint split = tg % p.splits;
    const uint hb = (tg / p.splits) % p.head_blocks;
    const uint item = (tg / p.splits) / p.head_blocks;
    const uint t = item / p.kv_head_count, kv = item % p.kv_head_count;
    const uint head0 = kv * all_heads + hb * kMaxGroup;  // first query head of this threadgroup
    const uint group = min(kMaxGroup, all_heads - hb * kMaxGroup);
    const uint visible = p.position0 + t + 1;
    const uint start = (p.window != 0 && visible > p.window) ? visible - p.window : 0;  // sliding window
    const uint pos0 = split * kSmallChunk, end = min(visible, pos0 + kSmallChunk);
    const uint first = max(pos0, start);  // first position this split scores
    device const float* keys = k_cache + (ulong)kv * p.max_context * p.head_dim;
    device const float* values = v_cache + (ulong)kv * p.max_context * p.head_dim;
    // Lane dimensions: 4 at lane * 4, and for 256-wide heads 4 more at 128 + lane * 4.
    const uint d4 = lane * 4;
    const bool has_dim = d4 < p.head_dim;

    float4 qv[8];  // slot g * kD4 + j
    for (uint g = 0; g < kMaxGroup; ++g) {
        for (uint j = 0; j < kD4; ++j) {
            qv[g * kD4 + j] = (g < group && has_dim)
                                  ? *(device const float4*)(q + (t * p.head_count + head0 + g) * p.head_dim + d4 + 128 * j)
                                  : float4(0.0f);
        }
    }

    // Scores: this simdgroup's positions are first + sg, + 4, + 8, ...
    for (uint pos = first + sg; pos < end; pos += 4) {
        float4 k[4];
        for (uint j = 0; j < kD4; ++j) k[j] = has_dim ? *(device const float4*)(keys + pos * p.head_dim + d4 + 128 * j) : float4(0.0f);
        for (uint g = 0; g < group; ++g) {
            float dotp = 0.0f;
            for (uint j = 0; j < kD4; ++j) dotp += dot(qv[g * kD4 + j], k[j]);
            const float s = simd_sum(dotp) * p.scale;
            if (lane == 0) sc[g * kSmallChunk + pos - pos0] = s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Softmax statistics per query over the split's scored positions (simdgroup sg handles queries sg, sg + 4).
    const uint lo = first < end ? first - pos0 : 0, count = first < end ? end - pos0 : 0;
    for (uint g = sg; g < group; g += 4) {
        float m = -INFINITY;
        for (uint i = lo + lane; i < count; i += 32) m = max(m, sc[g * kSmallChunk + i]);
        m = simd_max(m);
        float sum = 0.0f;
        for (uint i = lo + lane; i < count; i += 32) {
            const float e = exp(sc[g * kSmallChunk + i] - m);
            sc[g * kSmallChunk + i] = e;
            sum += e;
        }
        sum = simd_sum(sum);
        if (lane == 0) {
            stats[g][0] = m;
            stats[g][1] = sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Weighted values over the same positions, then the 4 simdgroups' sums are added.
    float4 acc[8];  // slot g * kD4 + j
    for (uint i = 0; i < 8; ++i) acc[i] = float4(0.0f);
    for (uint pos = first + sg; pos < end; pos += 4) {
        float4 v[4];
        for (uint j = 0; j < kD4; ++j) v[j] = has_dim ? *(device const float4*)(values + pos * p.head_dim + d4 + 128 * j) : float4(0.0f);
        for (uint g = 0; g < group; ++g) {
            const float w = sc[g * kSmallChunk + pos - pos0];
            for (uint j = 0; j < kD4; ++j) acc[g * kD4 + j] += w * v[j];
        }
    }
    for (uint g = 0; g < group; ++g) {
        for (uint j = 0; j < kD4; ++j) part[sg][g * kD4 + j][lane] = acc[g * kD4 + j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint g = sg; g < group; g += 4) {
        if (!has_dim) break;
        const ulong slot = ((ulong)t * p.head_count + head0 + g) * p.splits + split;
        device float* o = partial + slot * (p.head_dim + 2);
        for (uint j = 0; j < kD4; ++j) {
            const uint e = g * kD4 + j;
            const float4 total = part[0][e][lane] + part[1][e][lane] + part[2][e][lane] + part[3][e][lane];
            *(device float4*)(o + d4 + 128 * j) = total;
        }
        if (lane == 0) {
            o[p.head_dim] = stats[g][0];
            o[p.head_dim + 1] = stats[g][1];
        }
    }
}

// Merges the splits' partial softmax sums: one simdgroup per (token, head).
kernel void attention_combine(device const float* partial [[buffer(0)]],
                              device float* out [[buffer(1)]],
                              constant AttentionParams& p [[buffer(2)]],
                              device half* out_half [[buffer(3)]],
                              uint tg [[threadgroup_position_in_grid]],
                              uint sg [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
    const uint item = tg * 4 + sg;  // token * head_count + head
    if (item >= p.n * p.head_count) return;
    device const float* parts = partial + (ulong)item * p.splits * (p.head_dim + 2);
    float m = -INFINITY;
    for (uint s = 0; s < p.splits; ++s) m = max(m, parts[s * (p.head_dim + 2) + p.head_dim]);
    float denom = 0.0f;
    for (uint s = 0; s < p.splits; ++s) {
        const float ms = parts[s * (p.head_dim + 2) + p.head_dim];
        if (ms > -INFINITY) denom += parts[s * (p.head_dim + 2) + p.head_dim + 1] * exp(ms - m);
    }
    for (uint d = lane; d < p.head_dim; d += 32) {
        float acc = 0.0f;
        for (uint s = 0; s < p.splits; ++s) {
            const float ms = parts[s * (p.head_dim + 2) + p.head_dim];
            if (ms > -INFINITY) acc += parts[s * (p.head_dim + 2) + d] * exp(ms - m);
        }
        const float v = acc / denom;
        out[(ulong)item * p.head_dim + d] = v;
        if (p.write_half) out_half[(ulong)item * p.head_dim + d] = half(v);
    }
}
