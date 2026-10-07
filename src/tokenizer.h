#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "gguf.h"

namespace brisk {

using Token = std::int32_t;

// Which pre-tokenizer regex a vocabulary was built with. They differ only in
// how digits are grouped: Qwen2 takes one digit per word, Llama 3 up to three.
enum class PreTokenizer { Qwen2, Llama3 };

// Splits text into the words that byte-pair encoding is applied to, following
// the Qwen2 pre-tokenizer regex:
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
// or, for Llama 3, the same with \p{N}{1,3}.
// The returned views point into `text` and cover it completely, in order.
std::vector<std::string_view> pretokenize(std::string_view text, PreTokenizer pre);
inline std::vector<std::string_view> pretokenize_qwen2(std::string_view text) {
    return pretokenize(text, PreTokenizer::Qwen2);
}

// Byte-level BPE tokenizer for GGUF files with tokenizer.ggml.model == "gpt2"
// and the "qwen2" (Qwen2, Qwen2.5, Qwen3) or "llama-bpe" (Llama 3, BitNet
// b1.58 2B4T) pre-tokenizer.
class Tokenizer {
public:
    // Throws std::runtime_error if the file has no supported tokenizer.
    explicit Tokenizer(const GgufFile& file);

    // Internal views point into the object's own strings, so it cannot be copied.
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    // With parse_special, control tokens written out in the text (such as
    // "<|im_start|>") become their single token; without it they are treated
    // as plain text. No BOS or EOS token is added.
    std::vector<Token> encode(std::string_view text, bool parse_special = true) const;

    // The raw bytes of one token. A single token may be an incomplete UTF-8
    // sequence. Throws std::out_of_range for an invalid id.
    std::string_view piece(Token id) const;
    std::string decode(std::span<const Token> ids) const;

    std::size_t vocab_size() const { return pieces_.size(); }
    // The beginning-of-text token to put before a prompt, or -1 if the model uses none.
    Token bos() const { return bos_; }
    bool is_control(Token id) const;
    // True for tokens that should stop generation (end of turn or text).
    bool is_end_of_generation(Token id) const;

private:
    struct Special {
        std::string_view text;  // points into pieces_
        Token id;
        bool control;  // matched only when parse_special is set
    };
    struct Merge {
        std::uint32_t rank;
        std::uint32_t result;
    };

    std::uint32_t intern(const std::string& bytes);
    void encode_word(std::string_view word, std::vector<Token>& out) const;

    std::vector<std::string> pieces_;  // raw bytes per token id
    std::vector<std::int32_t> types_;
    std::vector<Special> specials_;    // longest text first
    std::vector<Token> end_tokens_;
    PreTokenizer pre_ = PreTokenizer::Qwen2;
    Token bos_ = -1;

    // BPE works on symbol ids. Ids below vocab_size() are token ids; higher
    // ids are merge operands or results that are not themselves tokens.
    std::unordered_map<std::string, std::uint32_t> symbol_ids_;
    std::vector<std::string> extra_symbols_;
    std::uint32_t byte_symbols_[256] = {};
    std::unordered_map<std::uint64_t, Merge> merges_;
};

}  // namespace brisk
