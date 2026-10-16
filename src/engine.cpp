#include "engine.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace brisk {

double Engine::log_likelihood(std::span<const Token> tokens) {
    if (tokens.size() < 2) throw std::runtime_error("need at least two tokens to score");
    // BRISK_EVAL_CHUNK caps the tokens scored per call, to exercise the
    // small-batch paths (1 for the gemv) with the perplexity check.
    std::size_t chunk = kMaxScoredTokens;
    if (const char* env = std::getenv("BRISK_EVAL_CHUNK"); env != nullptr) {
        chunk = std::clamp<std::size_t>(static_cast<std::size_t>(std::atoi(env)), 1, kMaxScoredTokens);
    }
    double total = 0.0;
    for (std::size_t begin = 0; begin < tokens.size(); begin += chunk) {
        const std::size_t n = std::min(chunk, tokens.size() - begin);
        const std::span<const float> logits = eval_all(tokens.subspan(begin, n));
        const std::size_t vocab = logits.size() / n;
        // Row t predicts tokens[begin + t + 1]; the last row of the text predicts nothing.
        for (std::size_t t = 0; t < n && begin + t + 1 < tokens.size(); ++t) {
            const float* row = logits.data() + t * vocab;
            const float max = *std::max_element(row, row + vocab);
            double sum = 0.0;
            for (std::size_t i = 0; i < vocab; ++i) sum += std::exp(static_cast<double>(row[i] - max));
            total += static_cast<double>(row[tokens[begin + t + 1]] - max) - std::log(sum);
        }
    }
    return total;
}

}  // namespace brisk
