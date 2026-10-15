#pragma once

#include <cstddef>
#include <span>

#include "tokenizer.h"

namespace brisk {

// One sequence being generated, on some device. Session (CPU) and
// MetalSession (Apple GPU) implement it; the rest of the program only sees
// this interface.
class Engine {
public:
    virtual ~Engine() = default;

    // Appends tokens to the sequence and returns the logits for the token
    // after the last one. The span is valid until the next call. Throws
    // std::runtime_error past the session's context length.
    virtual std::span<const float> eval(std::span<const Token> tokens) = 0;
    std::span<const float> eval(Token token) { return eval(std::span<const Token>(&token, 1)); }

    // Like eval(), but returns logits for every token: row i (vocab_size floats)
    // predicts the token after tokens[i]. At most kMaxScoredTokens tokens.
    virtual std::span<const float> eval_all(std::span<const Token> tokens) = 0;

    // Forgets everything from `position` on, so the next eval() continues there.
    virtual void rollback(std::size_t position) = 0;

    virtual std::size_t position() const = 0;

    static constexpr std::size_t kMaxScoredTokens = 64;

    // Appends tokens and returns the sum over i >= 1 of log p(tokens[i] | tokens[..i]),
    // the model's log-likelihood of the text. exp(-result / (n - 1)) is the perplexity.
    double log_likelihood(std::span<const Token> tokens);
};

}  // namespace brisk
