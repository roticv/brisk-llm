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

// Accumulates 32 dims per pass in registers: 8 vectors per head, so up to
// 2 heads at once; more heads take more passes.
void weighted_sum_f32(const float* values, std::size_t value_stride, std::size_t count, const float* weights,
                      std::size_t stride, std::size_t heads, std::size_t dim, float* out) {
    for (std::size_t h0 = 0; h0 < heads; h0 += 2) {
        const std::size_t nh = std::min<std::size_t>(2, heads - h0);
        const float* w0 = weights + h0 * stride;
        const float* w1 = weights + (h0 + nh - 1) * stride;  // same as w0 when nh == 1
        for (std::size_t d = 0; d < dim; d += 32) {
            float32x4_t a[8], b[8];
            for (int i = 0; i < 8; ++i) a[i] = b[i] = vdupq_n_f32(0.0f);
            for (std::size_t p = 0; p < count; ++p) {
                const float* v = values + p * value_stride + d;
                const float32x4_t s0 = vdupq_n_f32(w0[p]);
                const float32x4_t s1 = vdupq_n_f32(w1[p]);
                for (int i = 0; i < 8; ++i) {
                    const float32x4_t vv = vld1q_f32(v + 4 * i);
                    a[i] = vmlaq_f32(a[i], s0, vv);
                    b[i] = vmlaq_f32(b[i], s1, vv);
                }
            }
            for (int i = 0; i < 8; ++i) vst1q_f32(out + h0 * dim + d + 4 * static_cast<std::size_t>(i), a[i]);
            if (nh == 2) {
                for (int i = 0; i < 8; ++i) vst1q_f32(out + (h0 + 1) * dim + d + 4 * static_cast<std::size_t>(i), b[i]);
            }
        }
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

#endif

}  // namespace brisk::kernels
