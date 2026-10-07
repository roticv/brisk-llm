#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "model.h"
#include "tokenizer.h"

// Speculative decoding: a cheap drafter guesses the next few tokens, and the
// target model checks them all in one batched pass. The output is exactly
// what greedy decoding with the target alone would produce; the gain is
// that one pass over the target's weights yields several tokens when the
// guesses are right.

namespace brisk {

class Drafter {
public:
    virtual ~Drafter() = default;

    // Called once with the prompt, then after every round with the tokens
    // that were accepted plus the token the target chose next.
    virtual void observe(std::span<const Token> tokens) = 0;
    // Up to `count` guesses for the tokens following everything observed so far.
    virtual std::vector<Token> draft(std::size_t count) = 0;
};

// Guesses with a smaller model of the same tokenizer, decoding greedily.
class ModelDrafter : public Drafter {
public:
    ModelDrafter(const Model& model, std::size_t threads);
    void observe(std::span<const Token> tokens) override;
    std::vector<Token> draft(std::size_t count) override;

private:
    Session session_;
    std::vector<Token> pending_;  // observed but not yet fed to the model
};

// Guesses by finding the most recent earlier occurrence of the last few
// tokens and copying what followed it. Free, and effective on text that
// repeats itself (code, summaries, edits).
class LookupDrafter : public Drafter {
public:
    void observe(std::span<const Token> tokens) override;
    std::vector<Token> draft(std::size_t count) override;

private:
    std::vector<Token> history_;
};

struct SpeculativeStats {
    std::size_t rounds = 0;
    std::size_t drafted = 0;
    std::size_t accepted = 0;
};

class Speculator {
public:
    // `target` must already have evaluated the prompt except its last token,
    // which is `carry`; the drafter must have observed the whole prompt.
    Speculator(Session& target, Drafter& drafter, Token carry, std::size_t draft_tokens);

    // Runs one round and returns the tokens it settled: the accepted guesses
    // followed by the target's own next token. Never empty.
    std::vector<Token> step();

    const SpeculativeStats& stats() const { return stats_; }

private:
    Session& target_;
    Drafter& drafter_;
    Token carry_;
    std::size_t draft_tokens_;
    SpeculativeStats stats_;
    std::vector<Token> batch_;
};

}  // namespace brisk
