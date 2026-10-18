#include "kernels_f32.h"

#include <algorithm>
#include <cmath>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace brisk::kernels {

#if defined(__aarch64__)

namespace {

// exp(x) for four lanes, to about 2 ulp: x = n ln2 + r with |r| <= ln2/2,
// exp(r) by a degree-6 polynomial, then scale by 2^n through the exponent bits.
inline float32x4_t exp_f32x4(float32x4_t x) {
    x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-87.0f)), vdupq_n_f32(88.0f));
    const float32x4_t n = vrndnq_f32(vmulq_n_f32(x, 1.44269504088896341f));  // round(x / ln2)
    // r = x - n ln2, with ln2 split in two for precision.
    float32x4_t r = vmlsq_f32(x, n, vdupq_n_f32(0.693145751953125f));
    r = vmlsq_f32(r, n, vdupq_n_f32(1.428606765330187e-06f));
    float32x4_t p = vdupq_n_f32(1.0f / 720.0f);
    p = vmlaq_f32(vdupq_n_f32(1.0f / 120.0f), p, r);
    p = vmlaq_f32(vdupq_n_f32(1.0f / 24.0f), p, r);
    p = vmlaq_f32(vdupq_n_f32(1.0f / 6.0f), p, r);
    p = vmlaq_f32(vdupq_n_f32(0.5f), p, r);
    p = vmlaq_f32(vdupq_n_f32(1.0f), p, r);
    p = vmlaq_f32(vdupq_n_f32(1.0f), p, r);
    const int32x4_t exponent = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23);
    return vmulq_f32(p, vreinterpretq_f32_s32(exponent));
}

}  // namespace

float dot_f32(const float* a, const float* b, std::size_t n) {
    float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f), acc2 = vdupq_n_f32(0.0f), acc3 = vdupq_n_f32(0.0f);
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        acc0 = vmlaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
        acc1 = vmlaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        acc2 = vmlaq_f32(acc2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        acc3 = vmlaq_f32(acc3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    for (; i + 4 <= n; i += 4) acc0 = vmlaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
    float sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

void axpy_f32(float scale, const float* x, float* out, std::size_t n) {
    const float32x4_t s = vdupq_n_f32(scale);
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        vst1q_f32(out + i, vmlaq_f32(vld1q_f32(out + i), s, vld1q_f32(x + i)));
        vst1q_f32(out + i + 4, vmlaq_f32(vld1q_f32(out + i + 4), s, vld1q_f32(x + i + 4)));
        vst1q_f32(out + i + 8, vmlaq_f32(vld1q_f32(out + i + 8), s, vld1q_f32(x + i + 8)));
        vst1q_f32(out + i + 12, vmlaq_f32(vld1q_f32(out + i + 12), s, vld1q_f32(x + i + 12)));
    }
    for (; i + 4 <= n; i += 4) vst1q_f32(out + i, vmlaq_f32(vld1q_f32(out + i), s, vld1q_f32(x + i)));
    for (; i < n; ++i) out[i] += scale * x[i];
}

void softmax_f32(std::span<float> x) {
    const std::size_t n = x.size();
    float* p = x.data();
    float32x4_t vmax = vdupq_n_f32(-INFINITY);
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) vmax = vmaxq_f32(vmax, vld1q_f32(p + i));
    float max = vmaxvq_f32(vmax);
    for (; i < n; ++i) max = std::max(max, p[i]);

    const float32x4_t vm = vdupq_n_f32(max);
    float32x4_t vsum = vdupq_n_f32(0.0f);
    for (i = 0; i + 4 <= n; i += 4) {
        const float32x4_t e = exp_f32x4(vsubq_f32(vld1q_f32(p + i), vm));
        vst1q_f32(p + i, e);
        vsum = vaddq_f32(vsum, e);
    }
    double sum = vaddvq_f32(vsum);
    for (; i < n; ++i) {
        p[i] = std::exp(p[i] - max);
        sum += p[i];
    }
    const float scale = static_cast<float>(1.0 / sum);
    for (i = 0; i + 4 <= n; i += 4) vst1q_f32(p + i, vmulq_n_f32(vld1q_f32(p + i), scale));
    for (; i < n; ++i) p[i] *= scale;
}

namespace {

// 4 queries x 4 keys: 16 vector accumulators, each reduced once at the end.
inline void score_block_4x4(const float* const q[4], const float* const k[4], std::size_t dim, float scale, float* out[4],
                            std::size_t nq, std::size_t np) {
    float32x4_t acc[4][4];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) acc[i][j] = vdupq_n_f32(0.0f);
    }
    for (std::size_t d = 0; d < dim; d += 4) {
        const float32x4_t kv[4] = {vld1q_f32(k[0] + d), vld1q_f32(k[1] + d), vld1q_f32(k[2] + d), vld1q_f32(k[3] + d)};
        for (int i = 0; i < 4; ++i) {
            const float32x4_t qv = vld1q_f32(q[i] + d);
            for (int j = 0; j < 4; ++j) acc[i][j] = vmlaq_f32(acc[i][j], qv, kv[j]);
        }
    }
    for (std::size_t i = 0; i < nq; ++i) {
        for (std::size_t j = 0; j < np; ++j) out[i][j] = vaddvq_f32(acc[i][j]) * scale;
    }
}

}  // namespace

void scores_f32(const float* const* queries, std::size_t nq, const float* keys, std::size_t key_stride, std::size_t np,
                std::size_t dim, float scale, float* const* scores) {
    for (std::size_t q0 = 0; q0 < nq; q0 += 4) {
        const std::size_t qn = std::min<std::size_t>(4, nq - q0);
        // Missing queries/keys in a partial block repeat the block's last one; their results are discarded.
        const float* q[4];
        float* out[4];
        for (std::size_t i = 0; i < 4; ++i) {
            q[i] = queries[q0 + std::min(i, qn - 1)];
            out[i] = scores[q0 + std::min(i, qn - 1)];
        }
        for (std::size_t p0 = 0; p0 < np; p0 += 4) {
            const std::size_t pn = std::min<std::size_t>(4, np - p0);
            const float* k[4];
            for (std::size_t j = 0; j < 4; ++j) k[j] = keys + (p0 + std::min(j, pn - 1)) * key_stride;
            float* o[4];
            for (std::size_t i = 0; i < 4; ++i) o[i] = out[i] + p0;
            score_block_4x4(q, k, dim, scale, o, qn, pn);
        }
    }
}

// Positions are taken in chunks small enough that a chunk's value rows stay
// in L1 across the four 32-dim passes (on the A72, with 32 KB of L1, a pass
// over 500 positions would otherwise re-fetch every row from L2). Within a
// pass, 8 accumulator vectors per head, two heads at a time, live in
// registers; they are saved to `out` between chunks.
void weighted_sum_f32(const float* values, std::size_t value_stride, std::size_t count, const float* weights,
                      std::size_t stride, std::size_t heads, std::size_t dim, float* out) {
    constexpr std::size_t kPositionChunk = 32;
    std::fill_n(out, heads * dim, 0.0f);
    for (std::size_t p0 = 0; p0 < count; p0 += kPositionChunk) {
        const std::size_t p1 = std::min(count, p0 + kPositionChunk);
        for (std::size_t h0 = 0; h0 < heads; h0 += 2) {
            const std::size_t nh = std::min<std::size_t>(2, heads - h0);
            const float* w0 = weights + h0 * stride;
            const float* w1 = weights + (h0 + nh - 1) * stride;  // same as w0 when nh == 1
            float* o0 = out + h0 * dim;
            float* o1 = out + (h0 + nh - 1) * dim;
            for (std::size_t d = 0; d < dim; d += 32) {
                float32x4_t a[8], b[8];
                for (int i = 0; i < 8; ++i) {
                    a[i] = vld1q_f32(o0 + d + 4 * static_cast<std::size_t>(i));
                    b[i] = nh == 2 ? vld1q_f32(o1 + d + 4 * static_cast<std::size_t>(i)) : vdupq_n_f32(0.0f);
                }
                for (std::size_t p = p0; p < p1; ++p) {
                    const float* v = values + p * value_stride + d;
                    const float32x4_t s0 = vdupq_n_f32(w0[p]);
                    const float32x4_t s1 = vdupq_n_f32(w1[p]);
                    for (int i = 0; i < 8; ++i) {
                        const float32x4_t vv = vld1q_f32(v + 4 * i);
                        a[i] = vmlaq_f32(a[i], s0, vv);
                        b[i] = vmlaq_f32(b[i], s1, vv);
                    }
                }
                for (int i = 0; i < 8; ++i) vst1q_f32(o0 + d + 4 * static_cast<std::size_t>(i), a[i]);
                if (nh == 2) {
                    for (int i = 0; i < 8; ++i) vst1q_f32(o1 + d + 4 * static_cast<std::size_t>(i), b[i]);
                }
            }
        }
    }
}

void silu_mul_f32(float* gate, const float* up, std::size_t n) {
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const float32x4_t g = vld1q_f32(gate + i);
        // g / (1 + exp(-g)), with the division by Newton-refined reciprocal estimate.
        const float32x4_t denom = vaddq_f32(vdupq_n_f32(1.0f), exp_f32x4(vnegq_f32(g)));
        float32x4_t inv = vrecpeq_f32(denom);
        inv = vmulq_f32(inv, vrecpsq_f32(denom, inv));
        inv = vmulq_f32(inv, vrecpsq_f32(denom, inv));
        vst1q_f32(gate + i, vmulq_f32(vmulq_f32(g, inv), vld1q_f32(up + i)));
    }
    for (; i < n; ++i) gate[i] = gate[i] / (1.0f + std::exp(-gate[i])) * up[i];
}

void gelu_mul_f32(float* gate, const float* up, std::size_t n) {
    // 0.5 x (1 + tanh(z)) with z = sqrt(2/pi) (x + 0.044715 x^3), and
    // 1 + tanh(z) = 2 / (1 + exp(-2z)): gelu = x / (1 + exp(-2z)).
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const float32x4_t x = vld1q_f32(gate + i);
        const float32x4_t x3 = vmulq_f32(vmulq_f32(x, x), x);
        const float32x4_t z = vmulq_n_f32(vmlaq_n_f32(x, x3, 0.044715f), 0.7978845608f);
        const float32x4_t denom = vaddq_f32(vdupq_n_f32(1.0f), exp_f32x4(vmulq_n_f32(z, -2.0f)));
        float32x4_t inv = vrecpeq_f32(denom);
        inv = vmulq_f32(inv, vrecpsq_f32(denom, inv));
        inv = vmulq_f32(inv, vrecpsq_f32(denom, inv));
        vst1q_f32(gate + i, vmulq_f32(vmulq_f32(x, inv), vld1q_f32(up + i)));
    }
    for (; i < n; ++i) {
        const float x = gate[i];
        gate[i] = 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x))) * up[i];
    }
}

#else

float dot_f32(const float* a, const float* b, std::size_t n) {
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

void axpy_f32(float scale, const float* x, float* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) out[i] += scale * x[i];
}

void scores_f32(const float* const* queries, std::size_t nq, const float* keys, std::size_t key_stride, std::size_t np,
                std::size_t dim, float scale, float* const* scores) {
    for (std::size_t i = 0; i < nq; ++i) {
        for (std::size_t p = 0; p < np; ++p) scores[i][p] = dot_f32(queries[i], keys + p * key_stride, dim) * scale;
    }
}

void weighted_sum_f32(const float* values, std::size_t value_stride, std::size_t count, const float* weights,
                      std::size_t stride, std::size_t heads, std::size_t dim, float* out) {
    std::fill_n(out, heads * dim, 0.0f);
    for (std::size_t p = 0; p < count; ++p) {
        const float* v = values + p * value_stride;
        for (std::size_t h = 0; h < heads; ++h) axpy_f32(weights[h * stride + p], v, out + h * dim, dim);
    }
}

void softmax_f32(std::span<float> x) {
    const float max = *std::max_element(x.begin(), x.end());
    double sum = 0.0;
    for (float& v : x) {
        v = std::exp(v - max);
        sum += v;
    }
    const float scale = static_cast<float>(1.0 / sum);
    for (float& v : x) v *= scale;
}


void silu_mul_f32(float* gate, const float* up, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) gate[i] = gate[i] / (1.0f + std::exp(-gate[i])) * up[i];
}

void gelu_mul_f32(float* gate, const float* up, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        const float x = gate[i];
        gate[i] = 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x))) * up[i];
    }
}

#endif

}  // namespace brisk::kernels
