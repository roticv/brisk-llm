// Pre-tokenizer cases, with expected splits taken from llama.cpp's behaviour
// (tests/compare_tokenizer.py checks the full tokenizer against it on a
// corpus; these guard the splitting rules without needing a model file).

#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

#include "tokenizer.h"

namespace {

#define CHECK(cond)                                                                       \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            std::exit(1);                                                                 \
        }                                                                                 \
    } while (0)

void expect(std::string_view text, brisk::PreTokenizer pre, const std::vector<std::string_view>& expected) {
    const std::vector<std::string_view> got = brisk::pretokenize(text, pre);
    if (got != expected) {
        std::fprintf(stderr, "pretokenize(%s) gave:", std::string(text).c_str());
        for (const auto w : got) std::fprintf(stderr, " [%s]", std::string(w).c_str());
        std::fprintf(stderr, "\n");
        std::exit(1);
    }
}

}  // namespace

int main() {
    using brisk::PreTokenizer;
    const PreTokenizer qwen = PreTokenizer::Qwen2, llama = PreTokenizer::Llama3;

    expect("Hello, world!", qwen, {"Hello", ",", " world", "!"});
    expect("I'm sure they'll say it's fine", qwen, {"I", "'m", " sure", " they", "'ll", " say", " it", "'s", " fine"});
    expect("DON'T", qwen, {"DON", "'T"});
    expect("  two  spaces", qwen, {" ", " two", " ", " spaces"});
    expect("line one\nline two", qwen, {"line", " one", "\n", "line", " two"});
    expect("end  \n  next", qwen, {"end", "  \n", " ", " next"});
    expect("trailing ", qwen, {"trailing", " "});
    expect("x = 3.14 + 2000", qwen, {"x", " =", " ", "3", ".", "1", "4", " +", " ", "2", "0", "0", "0"});
    expect("x = 3.14 + 2000", llama, {"x", " =", " ", "3", ".", "14", " +", " ", "200", "0"});
    expect("naïve café", qwen, {"naïve", " café"});
    expect("你好,世界", qwen, {"你好", ",世界"});  // punctuation may prefix a run of letters
    expect("a b", qwen, {"a", " b"});  // no-break space is whitespace, so it prefixes the letters
    expect("", qwen, {});

    std::printf("tokenizer_test: ok\n");
    return 0;
}
