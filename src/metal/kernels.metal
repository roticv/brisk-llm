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
    uint paired;  // rows of w0 and w1 are paired (gate/up) and combined by the epilogue
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
    uint splits;      // position ranges per (token block, kv head); > 1 writes partials for attention_combine
    uint chunk;       // positions per split, a multiple of 8
    uint block_tokens;  // tokens per threadgroup of the batched kernel: kMaxQueries / group
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
// turn (both rows' blocks per lane, which keeps more loads in flight) and
// the partial sums are reduced across the simdgroup.
// The 4 simdgroups of a threadgroup take consecutive row pairs, numbered
// across the three matrices in turn. With `paired`, the pair is instead row
// r of w0 and row r of w1 (the gate and up projections) and the epilogue
// combines them: out0[r] = f(gate) * up.
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
    const uint pairs0 = paired ? p.rows[0] : (p.rows[0] + 1) / 2;
    const uint pairs1 = paired ? 0 : (p.rows[1] + 1) / 2, pairs2 = paired ? 0 : (p.rows[2] + 1) / 2;
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
    const uint blocks = p.cols / 32;
    const uint row0 = paired ? pair : pair * 2;
    const bool two = paired || row0 + 1 < rows;
    device const uchar* wrow0 = w + (ulong)row0 * blocks * 18;
    device const uchar* wrow1 = paired ? w1 + (ulong)row0 * blocks * 18 : wrow0 + (two ? blocks * 18 : 0);
    device const float4* xrow = (device const float4*)(x + t * p.cols);

    float acc0 = 0.0f, acc1 = 0.0f;
    for (uint b = lane; b < blocks; b += 32) {
        float4 va[8], vb[8];
        q4_block_dequant((device const Q4Block*)(wrow0 + b * 18), va);
        q4_block_dequant((device const Q4Block*)(wrow1 + b * 18), vb);
        float4 s0 = 0.0f, s1 = 0.0f;
        for (uint i = 0; i < 8; ++i) {
            const float4 xv = xrow[b * 8 + i];
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
            out[t * rows + row0] = epilogue(p.mode, acc1, acc0);  // acc0 is the gate, acc1 the up
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
// The tile's weight format: 0 = Q4_0, 1 = TQ2_0 (a function constant, so each is its own pipeline).
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
    device const uchar* wrow = w + (ulong)(row0 + r) * (ternary ? p.cols / 256 * 66 : blocks * 18);
    device const half4* xsrc = (device const half4*)(x + (t0 + (valid ? t : 0)) * p.cols + k0);
    threadgroup half* wdst = sa + 64 * ((high * 2) * 8 + r / 8) + r % 8;  // k = high * 16, then + 8 per block
    threadgroup half4* xdst = (threadgroup half4*)(sb + 64 * ((t / 8) * 4 + k0 / 8) + 8 * (t % 8));
    const uint shift = high * 4;

    half wd;
    ushort wq[8];
    half4 xv[4];
    uint tl = 0;  // ternary: the bit pair of this step's 32 columns
    auto load = [&](uint b) {
        if (ternary) {
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
            const half4 v = ternary ? (half4(extract_bits(uint4(wv) >> uint4(0, 8, 16, 24), 2 * tl, 2)) - 1.0h) * wd
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
constant uint kMaxQueries = 16;  // tokens per block * query heads per key/value head (p.block_tokens * group)

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
    threadgroup float row_max[kMaxQueries];
    const uint kv_heads = p.kv_head_count;
    const uint group = p.head_count / kv_heads;  // query heads per key/value head
    const uint split = tg % p.splits;
    const uint block = (tg / p.splits) / kv_heads, kv = (tg / p.splits) % kv_heads;
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
    for (uint pos = pos0 + sg * 8; pos < visible_last; pos += 32) {
        simdgroup_float8x8 acc0(0.0f), acc1(0.0f);
        for (uint d = 0; d < p.head_dim; d += 8) {
            simdgroup_float8x8 kb, q0, q1;
            simdgroup_load(kb, keys + pos * p.head_dim + d, p.head_dim, ulong2(0, 0), true);  // transposed: [d][pos]
            simdgroup_load(q0, qs + d, 128);
            simdgroup_load(q1, qs + 8 * 128 + d, 128);
            simdgroup_multiply_accumulate(acc0, q0, kb, acc0);
            simdgroup_multiply_accumulate(acc1, q1, kb, acc1);
        }
        simdgroup_store(acc0, sc + pos, row_stride);
        simdgroup_store(acc1, sc + 8 * row_stride + pos, row_stride);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Softmax per query row over the positions its token sees; the rest of
    // the row up to the padded end is zeroed (rows past `queries` too) so the
    // value pass can multiply whole 8-position blocks.
    const uint padded = (visible_last + 7) / 8 * 8;
    for (uint qi = sg; qi < kMaxQueries; qi += 4) {
        const uint visible = qi < queries ? min(p.position0 + t0 + qi / group + 1, visible_last) : 0;
        device float* row = sc + qi * row_stride;
        float m = -INFINITY;
        for (uint pos = pos0 + lane; pos < visible; pos += 32) m = max(m, row[pos]);
        m = simd_max(m) * p.scale;  // the row's maximum in scaled units (-inf when it sees nothing here)
        float sum = 0.0f;
        for (uint pos = pos0 + lane; pos < padded; pos += 32) {
            const float e = pos < visible ? exp(row[pos] * p.scale - m) : 0.0f;
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
    // 32-dimension slice; rows past `queries` and positions past
    // `visible_last` carry zero weights.
    {
        const uint d0 = sg * 32;
        simdgroup_float8x8 acc[2][4];
        for (uint i = 0; i < 2; ++i) {
            for (uint j = 0; j < 4; ++j) acc[i][j] = simdgroup_float8x8(0.0f);
        }
        if (d0 < p.head_dim) {
            for (uint pos = pos0; pos < visible_last; pos += 8) {
                simdgroup_float8x8 pb[2], vb[4];
                simdgroup_load(pb[0], sc + pos, row_stride);
                simdgroup_load(pb[1], sc + 8 * row_stride + pos, row_stride);
                for (uint j = 0; j < 4; ++j) simdgroup_load(vb[j], values + pos * p.head_dim + d0 + j * 8, p.head_dim);
                for (uint i = 0; i < 2; ++i) {
                    for (uint j = 0; j < 4; ++j) simdgroup_multiply_accumulate(acc[i][j], pb[i], vb[j], acc[i][j]);
                }
            }
        }
        // Stage [query][dim slice] and write the queries that exist, normalised.
        threadgroup float stage[kMaxQueries * 128];
        for (uint i = 0; i < 2; ++i) {
            for (uint j = 0; j < 4; ++j) simdgroup_store(acc[i][j], stage + (8 * i) * 128 + d0 + j * 8, 128);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (p.splits > 1) {
            // Partial result for attention_combine: unnormalised sums with the
            // row's maximum and denominator, at [token][head][split].
            for (uint i = tid; i < queries * p.head_dim; i += 128) {
                const uint qi = i / p.head_dim, d = i % p.head_dim;
                const uint t = t0 + qi / group, h = kv * group + qi % group;
                const ulong slot = ((ulong)t * p.head_count + h) * p.splits + split;
                out[slot * (p.head_dim + 2) + d] = stage[qi * 128 + d];
                if (d == 0) {
                    out[slot * (p.head_dim + 2) + p.head_dim] = row_max[qi];
                    out[slot * (p.head_dim + 2) + p.head_dim + 1] = inv_sum[qi] > 0.0f ? 1.0f / inv_sum[qi] : 0.0f;
                }
            }
            return;
        }
        for (uint i = tid; i < queries * p.head_dim; i += 128) {
            const uint qi = i / p.head_dim, d = i % p.head_dim;
            const uint t = t0 + qi / group, h = kv * group + qi % group;
            const float v = stage[qi * 128 + d] * inv_sum[qi];
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
constant uint kMaxGroup = 4;     // query heads per key/value head

kernel void attention_small(device const float* q [[buffer(0)]],
                            device const float* k_cache [[buffer(1)]],
                            device const float* v_cache [[buffer(2)]],
                            device float* partial [[buffer(3)]],
                            constant AttentionParams& p [[buffer(4)]],
                            uint tg [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sc[kMaxGroup * kSmallChunk];
    threadgroup float4 part[4][kMaxGroup][32];  // per-simdgroup value sums, lanes' float4s
    threadgroup float stats[kMaxGroup][2];      // max, sum per query
    const uint group = p.head_count / p.kv_head_count;
    const uint split = tg % p.splits;
    const uint t = (tg / p.splits) / p.kv_head_count, kv = (tg / p.splits) % p.kv_head_count;
    const uint visible = p.position0 + t + 1;
    const uint pos0 = split * kSmallChunk, end = min(visible, pos0 + kSmallChunk);
    device const float* keys = k_cache + (ulong)kv * p.max_context * p.head_dim;
    device const float* values = v_cache + (ulong)kv * p.max_context * p.head_dim;
    const uint d4 = lane * 4;
    const bool has_dim = d4 < p.head_dim;

    float4 qv[kMaxGroup];
    for (uint g = 0; g < kMaxGroup; ++g) {
        qv[g] = (g < group && has_dim) ? *(device const float4*)(q + (t * p.head_count + kv * group + g) * p.head_dim + d4)
                                       : float4(0.0f);
    }

    // Scores: this simdgroup's positions are pos0 + sg, + 4, + 8, ...
    for (uint pos = pos0 + sg; pos < end; pos += 4) {
        const float4 k = has_dim ? *(device const float4*)(keys + pos * p.head_dim + d4) : float4(0.0f);
        for (uint g = 0; g < group; ++g) {
            const float s = simd_sum(dot(qv[g], k)) * p.scale;
            if (lane == 0) sc[g * kSmallChunk + pos - pos0] = s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Softmax statistics per query over the split (simdgroup g handles query g).
    const uint count = end > pos0 ? end - pos0 : 0;
    if (sg < group) {
        float m = -INFINITY;
        for (uint i = lane; i < count; i += 32) m = max(m, sc[sg * kSmallChunk + i]);
        m = simd_max(m);
        float sum = 0.0f;
        for (uint i = lane; i < count; i += 32) {
            const float e = exp(sc[sg * kSmallChunk + i] - m);
            sc[sg * kSmallChunk + i] = e;
            sum += e;
        }
        sum = simd_sum(sum);
        if (lane == 0) {
            stats[sg][0] = m;
            stats[sg][1] = sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Weighted values over the same positions, then the 4 simdgroups' sums are added.
    float4 acc[kMaxGroup];
    for (uint g = 0; g < kMaxGroup; ++g) acc[g] = float4(0.0f);
    for (uint pos = pos0 + sg; pos < end; pos += 4) {
        const float4 v = has_dim ? *(device const float4*)(values + pos * p.head_dim + d4) : float4(0.0f);
        for (uint g = 0; g < group; ++g) acc[g] += sc[g * kSmallChunk + pos - pos0] * v;
    }
    for (uint g = 0; g < group; ++g) part[sg][g][lane] = acc[g];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg < group && has_dim) {
        const float4 total = part[0][sg][lane] + part[1][sg][lane] + part[2][sg][lane] + part[3][sg][lane];
        const ulong slot = ((ulong)t * p.head_count + kv * group + sg) * p.splits + split;
        device float* o = partial + slot * (p.head_dim + 2);
        *(device float4*)(o + d4) = total;
        if (lane == 0) {
            o[p.head_dim] = stats[sg][0];
            o[p.head_dim + 1] = stats[sg][1];
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
