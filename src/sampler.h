#pragma once

#include <cstdint>
#include <random>
#include <span>
#include <vector>

#include "tokenizer.h"

namespace brisk {

struct SamplerConfig {
    float temperature = 0.0f;  // 0 or below picks the most likely token every time
    std::uint32_t top_k = 20;  // keep only the k most likely tokens; 0 keeps all
    float top_p = 0.95f;       // then keep the smallest set whose probability reaches p
    std::uint64_t seed = 0;
};

class Sampler {
public:
    explicit Sampler(const SamplerConfig& config);

    // `logits` must not be empty.
    Token sample(std::span<const float> logits);

private:
    struct Candidate {
        Token id;
        float value;  // logit, then probability
    };

    SamplerConfig config_;
    std::mt19937_64 rng_;
    std::vector<Candidate> candidates_;
};

}  // namespace brisk
