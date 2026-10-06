#include "sampler.h"

#include <algorithm>
#include <cmath>

namespace brisk {

Sampler::Sampler(const SamplerConfig& config) : config_(config), rng_(config.seed) {}

Token Sampler::sample(std::span<const float> logits) {
    if (config_.temperature <= 0.0f) {
        return static_cast<Token>(std::max_element(logits.begin(), logits.end()) - logits.begin());
    }

    candidates_.resize(logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i) candidates_[i] = {static_cast<Token>(i), logits[i]};

    const auto more_likely = [](const Candidate& a, const Candidate& b) { return a.value > b.value; };
    std::size_t keep = candidates_.size();
    if (config_.top_k > 0 && config_.top_k < keep) keep = config_.top_k;
    std::partial_sort(candidates_.begin(), candidates_.begin() + static_cast<std::ptrdiff_t>(keep), candidates_.end(),
                      more_likely);
    candidates_.resize(keep);

    // Softmax at the given temperature; candidates are sorted, so the first is the maximum.
    const float max = candidates_.front().value;
    double total = 0.0;
    for (Candidate& c : candidates_) {
        c.value = std::exp((c.value - max) / config_.temperature);
        total += c.value;
    }

    // Top-p: cut the tail once the kept probability mass reaches p.
    double kept = 0.0;
    std::size_t count = 0;
    while (count < candidates_.size()) {
        kept += candidates_[count++].value;
        if (kept >= static_cast<double>(config_.top_p) * total) break;
    }

    double target = std::uniform_real_distribution<double>(0.0, kept)(rng_);
    for (std::size_t i = 0; i < count; ++i) {
        target -= candidates_[i].value;
        if (target < 0.0) return candidates_[i].id;
    }
    return candidates_[count - 1].id;
}

}  // namespace brisk
