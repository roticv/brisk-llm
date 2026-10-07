#include "tokenizer.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

#include "unicode.h"

namespace brisk {

namespace {

// Token types as stored in tokenizer.ggml.token_type.
constexpr std::int32_t kTypeNormal = 1;
constexpr std::int32_t kTypeUnknown = 2;
constexpr std::int32_t kTypeControl = 3;
constexpr std::int32_t kTypeUserDefined = 4;

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("tokenizer: " + what); }

bool is_special_type(std::int32_t type) {
    return type == kTypeControl || type == kTypeUserDefined || type == kTypeUnknown;
}

// GPT-2 style vocabularies store each raw byte as a printable character:
// bytes that are printable in Latin-1 map to themselves, the rest to
// U+0100 onwards. This inverts that mapping; -1 means "not a mapped byte".
struct ByteDecoder {
    int table[0x100 + 0x100];

    ByteDecoder() {
        std::fill(std::begin(table), std::end(table), -1);
        std::uint32_t next = 0x100;
        for (int b = 0; b < 0x100; ++b) {
            const bool printable = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
            table[printable ? static_cast<std::uint32_t>(b) : next++] = b;
        }
    }

    std::string operator()(std::string_view mapped) const {
        std::string out;
        out.reserve(mapped.size());
        for (const Codepoint& cp : decode_utf8(mapped)) {
            if (cp.valid && cp.value < std::size(table) && table[cp.value] >= 0) {
                out.push_back(static_cast<char>(table[cp.value]));
            } else {
                out.append(mapped.substr(cp.offset, cp.length));
            }
        }
        return out;
    }
};

std::uint64_t pair_key(std::uint32_t left, std::uint32_t right) {
    return (static_cast<std::uint64_t>(left) << 32) | right;
}

std::uint32_t ascii_lower(std::uint32_t cp) { return cp >= 'A' && cp <= 'Z' ? cp + ('a' - 'A') : cp; }

}  // namespace

// This mirrors llama.cpp's hand-written matcher for the same regex, including
// its treatment of edge cases, so both engines split text identically.
std::vector<std::string_view> pretokenize(std::string_view text, PreTokenizer pre) {
    constexpr std::uint32_t kOutOfRange = 0xFFFFFFFF;
    struct Flags {
        bool present = false;
        bool letter = false;
        bool number = false;
        bool whitespace = false;
        bool other() const { return present && !letter && !number && !whitespace; }
    };

    const std::vector<Codepoint> cps = decode_utf8(text);
    const std::size_t n = cps.size();

    // Invalid bytes classify as U+FFFD (a symbol) but keep their original bytes.
    auto cp_at = [&](std::size_t pos) -> std::uint32_t {
        if (pos >= n) return kOutOfRange;
        return cps[pos].valid ? cps[pos].value : 0xFFFD;
    };
    auto flags_at = [&](std::size_t pos) -> Flags {
        if (pos >= n) return {};
        const std::uint32_t cp = cp_at(pos);
        return {true, is_letter(cp), is_number(cp), is_whitespace(cp)};
    };
    auto is_newline = [](std::uint32_t cp) { return cp == '\r' || cp == '\n'; };

    std::vector<std::string_view> words;
    auto emit = [&](std::size_t begin, std::size_t end) {
        const std::size_t from = cps[begin].offset;
        const std::size_t to = end < n ? cps[end].offset : text.size();
        words.push_back(text.substr(from, to - from));
    };

    std::size_t pos = 0;
    while (pos < n) {
        const std::size_t begin = pos;
        const std::uint32_t cp = cp_at(pos);
        const Flags flags = flags_at(pos);

        // (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (cp == '\'' && pos + 1 < n) {
            const std::uint32_t next = ascii_lower(cp_at(pos + 1));
            if (next == 's' || next == 't' || next == 'm' || next == 'd') {
                pos += 2;
                emit(begin, pos);
                continue;
            }
            if (pos + 2 < n) {
                const std::uint32_t after = ascii_lower(cp_at(pos + 2));
                if ((next == 'r' && after == 'e') || (next == 'v' && after == 'e') || (next == 'l' && after == 'l')) {
                    pos += 3;
                    emit(begin, pos);
                    continue;
                }
            }
        }

        // [^\r\n\p{L}\p{N}]?\p{L}+
        if (!is_newline(cp) && !flags.number && (flags.letter || flags_at(pos + 1).letter)) {
            ++pos;
            while (flags_at(pos).letter) ++pos;
            emit(begin, pos);
            continue;
        }

        // \p{N} (Qwen2) or \p{N}{1,3} (Llama 3)
        if (flags.number) {
            ++pos;
            if (pre == PreTokenizer::Llama3) {
                while (flags_at(pos).number && pos - begin < 3) ++pos;
            }
            emit(begin, pos);
            continue;
        }

        //  ?[^\s\p{L}\p{N}]+[\r\n]*
        Flags run = cp == ' ' ? flags_at(pos + 1) : flags;
        if (!run.letter && !run.number && !run.whitespace) {
            if (cp == ' ') ++pos;
            while (run.other()) run = flags_at(++pos);
            while (is_newline(cp_at(pos))) ++pos;
            emit(begin, pos);
            continue;
        }

        std::size_t whitespace = 0;
        std::size_t newline_end = 0;
        while (flags_at(pos + whitespace).whitespace) {
            if (is_newline(cp_at(pos + whitespace))) newline_end = pos + whitespace + 1;
            ++whitespace;
        }

        // \s*[\r\n]+
        if (newline_end > 0) {
            pos = newline_end;
            emit(begin, pos);
            continue;
        }
        // \s+(?!\S): leave the last whitespace character to join the next word
        if (whitespace > 1 && pos + whitespace < n) {
            pos += whitespace - 1;
            emit(begin, pos);
            continue;
        }
        // \s+
        if (whitespace > 0) {
            pos += whitespace;
            emit(begin, pos);
            continue;
        }

        ++pos;
        emit(begin, pos);
    }
    return words;
}

Tokenizer::Tokenizer(const GgufFile& file) {
    const auto model = file.get<std::string_view>("tokenizer.ggml.model");
    const auto pre = file.get<std::string_view>("tokenizer.ggml.pre");
    if (model != "gpt2") fail("unsupported tokenizer model '" + std::string(model.value_or("(none)")) + "'");
    if (pre == "qwen2") pre_ = PreTokenizer::Qwen2;
    else if (pre == "llama-bpe") pre_ = PreTokenizer::Llama3;
    else fail("unsupported pre-tokenizer '" + std::string(pre.value_or("(none)")) + "'");

    const auto tokens = file.get<GgufArray>("tokenizer.ggml.tokens");
    if (!tokens) fail("missing tokenizer.ggml.tokens");
    const std::vector<std::string_view> mapped = tokens->strings();
    if (mapped.empty()) fail("empty vocabulary");
    if (mapped.size() > static_cast<std::size_t>(std::numeric_limits<Token>::max())) fail("vocabulary too large");

    if (const auto types = file.get<GgufArray>("tokenizer.ggml.token_type")) {
        types_ = types->values<std::int32_t>();
        if (types_.size() != mapped.size()) fail("token_type length does not match the vocabulary");
    } else {
        types_.assign(mapped.size(), kTypeNormal);
    }

    const ByteDecoder decode_bytes;
    pieces_.reserve(mapped.size());
    for (std::size_t id = 0; id < mapped.size(); ++id) {
        // Special tokens are literal text; everything else is byte-mapped.
        if (is_special_type(types_[id])) {
            pieces_.emplace_back(mapped[id]);
        } else {
            pieces_.push_back(decode_bytes(mapped[id]));
            symbol_ids_[pieces_.back()] = static_cast<std::uint32_t>(id);
        }
    }

    for (std::size_t id = 0; id < pieces_.size(); ++id) {
        if (is_special_type(types_[id]) && !pieces_[id].empty()) {
            specials_.push_back({pieces_[id], static_cast<Token>(id), types_[id] != kTypeUserDefined});
        }
    }
    std::stable_sort(specials_.begin(), specials_.end(),
                     [](const Special& a, const Special& b) { return a.text.size() > b.text.size(); });

    for (int b = 0; b < 0x100; ++b) byte_symbols_[b] = intern(std::string(1, static_cast<char>(b)));

    if (const auto merges = file.get<GgufArray>("tokenizer.ggml.merges")) {
        std::uint32_t rank = 0;
        for (const std::string_view line : merges->strings()) {
            const std::size_t space = line.find(' ', 1);
            if (space == std::string_view::npos) fail("malformed merge rule");
            const std::string left = decode_bytes(line.substr(0, space));
            const std::string right = decode_bytes(line.substr(space + 1));
            const std::uint32_t left_id = intern(left);
            const std::uint32_t right_id = intern(right);
            const std::uint32_t result = intern(left + right);
            merges_.try_emplace(pair_key(left_id, right_id), Merge{rank, result});  // first rule wins
            ++rank;
        }
    }

    auto add_end_token = [&](Token id) {
        if (id >= 0 && static_cast<std::size_t>(id) < pieces_.size() &&
            std::find(end_tokens_.begin(), end_tokens_.end(), id) == end_tokens_.end()) {
            end_tokens_.push_back(id);
        }
    };
    if (const auto eos = file.get<std::uint32_t>("tokenizer.ggml.eos_token_id")) add_end_token(static_cast<Token>(*eos));
    for (const Special& special : specials_) {
        if (special.text == "<|endoftext|>" || special.text == "<|im_end|>" || special.text == "<|eot_id|>" ||
            special.text == "<|end_of_text|>") {
            add_end_token(special.id);
        }
    }
    if (file.get<bool>("tokenizer.ggml.add_bos_token").value_or(false)) {
        if (const auto bos = file.get<std::uint32_t>("tokenizer.ggml.bos_token_id")) bos_ = static_cast<Token>(*bos);
    }
}

std::uint32_t Tokenizer::intern(const std::string& bytes) {
    const auto next = static_cast<std::uint32_t>(pieces_.size() + extra_symbols_.size());
    const auto [it, inserted] = symbol_ids_.try_emplace(bytes, next);
    if (inserted) extra_symbols_.push_back(bytes);
    return it->second;
}

std::string_view Tokenizer::piece(Token id) const {
    if (id < 0 || static_cast<std::size_t>(id) >= pieces_.size()) throw std::out_of_range("tokenizer: invalid token id");
    return pieces_[static_cast<std::size_t>(id)];
}

std::string Tokenizer::decode(std::span<const Token> ids) const {
    std::string out;
    for (const Token id : ids) out.append(piece(id));
    return out;
}

bool Tokenizer::is_control(Token id) const {
    return id >= 0 && static_cast<std::size_t>(id) < types_.size() && types_[static_cast<std::size_t>(id)] == kTypeControl;
}

bool Tokenizer::is_end_of_generation(Token id) const {
    return std::find(end_tokens_.begin(), end_tokens_.end(), id) != end_tokens_.end();
}

void Tokenizer::encode_word(std::string_view word, std::vector<Token>& out) const {
    std::vector<std::uint32_t> symbols;
    symbols.reserve(word.size());
    for (const char c : word) symbols.push_back(byte_symbols_[static_cast<unsigned char>(c)]);

    // Repeatedly apply the best-ranked merge, leftmost first on ties.
    while (symbols.size() > 1) {
        std::uint32_t best_rank = std::numeric_limits<std::uint32_t>::max();
        std::uint32_t best_result = 0;
        std::size_t best_pos = 0;
        for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
            const auto it = merges_.find(pair_key(symbols[i], symbols[i + 1]));
            if (it != merges_.end() && it->second.rank < best_rank) {
                best_rank = it->second.rank;
                best_result = it->second.result;
                best_pos = i;
            }
        }
        if (best_rank == std::numeric_limits<std::uint32_t>::max()) break;
        symbols[best_pos] = best_result;
        symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best_pos) + 1);
    }

    const auto vocab = static_cast<std::uint32_t>(pieces_.size());
    for (const std::uint32_t symbol : symbols) {
        if (symbol < vocab) {
            out.push_back(static_cast<Token>(symbol));
            continue;
        }
        // Not a token on its own: fall back to its bytes, dropping any that have no token.
        for (const char c : extra_symbols_[symbol - vocab]) {
            const std::uint32_t byte_symbol = byte_symbols_[static_cast<unsigned char>(c)];
            if (byte_symbol < vocab) out.push_back(static_cast<Token>(byte_symbol));
        }
    }
}

std::vector<Token> Tokenizer::encode(std::string_view text, bool parse_special) const {
    struct Fragment {
        std::string_view text;
        Token token = -1;  // >= 0 when this fragment is a special token
    };

    // Cut out special tokens, longest first, leaving plain text between them.
    std::vector<Fragment> fragments{{text}};
    for (const Special& special : specials_) {
        if (special.control && !parse_special) continue;
        std::vector<Fragment> next;
        bool found = false;
        for (const Fragment& fragment : fragments) {
            if (fragment.token >= 0) {
                next.push_back(fragment);
                continue;
            }
            std::string_view rest = fragment.text;
            for (std::size_t at = rest.find(special.text); at != std::string_view::npos; at = rest.find(special.text)) {
                if (at > 0) next.push_back({rest.substr(0, at)});
                next.push_back({special.text, special.id});
                rest.remove_prefix(at + special.text.size());
                found = true;
            }
            if (!rest.empty()) next.push_back({rest});
        }
        if (found) fragments = std::move(next);
    }

    std::vector<Token> out;
    for (const Fragment& fragment : fragments) {
        if (fragment.token >= 0) {
            out.push_back(fragment.token);
            continue;
        }
        for (const std::string_view word : pretokenize(fragment.text, pre_)) encode_word(word, out);
    }
    return out;
}

}  // namespace brisk
