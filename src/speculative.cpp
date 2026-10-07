#include "speculative.h"

#include <algorithm>

namespace brisk {

namespace {

Token argmax(std::span<const float> logits) {
    return static_cast<Token>(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

}  // namespace

// ---- ModelDrafter -----------------------------------------------------------

ModelDrafter::ModelDrafter(const Model& model, std::size_t threads) : session_(model, threads) {}

void ModelDrafter::observe(std::span<const Token> tokens) {
    // The draft model is kept exactly one token behind what was observed, so
    // its next eval yields the logits for the first guess.
    pending_.insert(pending_.end(), tokens.begin(), tokens.end());
}

std::vector<Token> ModelDrafter::draft(std::size_t count) {
    // Feed everything observed since last time; the guesses made then were
    // rolled back, so the model is in step with the target.
    std::span<const float> logits = session_.eval(pending_);
    pending_.clear();
    const std::size_t base = session_.position();

    std::vector<Token> guesses;
    for (std::size_t i = 0; i < count; ++i) {
        const Token next = argmax(logits);
        guesses.push_back(next);
        if (i + 1 < count) logits = session_.eval(next);
    }
    session_.rollback(base);  // the guesses are not part of the sequence until accepted
    return guesses;
}

// ---- LookupDrafter ----------------------------------------------------------

void LookupDrafter::observe(std::span<const Token> tokens) {
    history_.insert(history_.end(), tokens.begin(), tokens.end());
}

std::vector<Token> LookupDrafter::draft(std::size_t count) {
    constexpr std::size_t kMaxNgram = 3;
    const std::size_t n = history_.size();
    for (std::size_t ngram = std::min(kMaxNgram, n); ngram >= 1; --ngram) {
        const Token* suffix = history_.data() + n - ngram;
        // Most recent earlier occurrence of the suffix, with something after it.
        for (std::size_t start = n - ngram; start-- > 0;) {
            if (std::equal(suffix, suffix + ngram, history_.data() + start)) {
                const std::size_t from = start + ngram;
                const std::size_t available = std::min(count, n - ngram - from);
                if (available == 0) break;
                return {history_.begin() + static_cast<std::ptrdiff_t>(from),
                        history_.begin() + static_cast<std::ptrdiff_t>(from + available)};
            }
        }
    }
    return {};
}

// ---- Speculator -------------------------------------------------------------

Speculator::Speculator(Session& target, Drafter& drafter, Token carry, std::size_t draft_tokens)
    : target_(target), drafter_(drafter), carry_(carry), draft_tokens_(std::min(draft_tokens, Session::kMaxScoredTokens - 1)) {}

std::vector<Token> Speculator::step() {
    const std::vector<Token> guesses = drafter_.draft(draft_tokens_);
    const std::size_t base = target_.position();  // the carry token goes here

    batch_.clear();
    batch_.push_back(carry_);
    batch_.insert(batch_.end(), guesses.begin(), guesses.end());
    const std::span<const float> logits = target_.eval_all(batch_);
    const std::size_t vocab = logits.size() / batch_.size();

    // Row i predicts the token after batch_[i]; guess i+1 is right if row i picks it.
    std::vector<Token> settled;
    std::size_t accepted = 0;
    Token next = argmax(logits.subspan(0, vocab));
    while (accepted < guesses.size() && guesses[accepted] == next) {
        settled.push_back(next);
        ++accepted;
        next = argmax(logits.subspan(accepted * vocab, vocab));
    }
    settled.push_back(next);  // the target's own choice after the accepted run

    // Keep only carry + accepted guesses in the cache; `next` becomes the new carry.
    target_.rollback(base + 1 + accepted);
    drafter_.observe(settled);
    carry_ = next;

    stats_.rounds += 1;
    stats_.drafted += guesses.size();
    stats_.accepted += accepted;
    return settled;
}

}  // namespace brisk
