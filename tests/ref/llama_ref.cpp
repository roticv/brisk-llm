// Reference outputs from llama.cpp, used to check brisk-llm for correctness.
// Built by tests/ref/build.sh against the checkout in baselines/llama.cpp.
//
//   llama_ref tokenize <model.gguf> <strings.bin>
//       strings.bin holds records of [u32 length][bytes]. Prints one line of
//       space-separated token ids per record.
//   llama_ref eval <model.gguf> <prompt.txt> <n_generate> <logits.bin>
//       Prints the prompt's token ids, then n_generate greedily chosen ids.
//       Writes the float32 logits that follow the prompt to logits.bin.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "llama.h"

namespace {

constexpr int kThreads = 4;
constexpr std::uint32_t kContext = 2048;

std::string read_file(const char* path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", path);
        std::exit(1);
    }
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::vector<llama_token> tokenize(const llama_vocab* vocab, const std::string& text) {
    std::vector<llama_token> tokens(text.size() + 16);
    const int n = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()), tokens.data(),
                                 static_cast<int32_t>(tokens.size()), /*add_special=*/false, /*parse_special=*/true);
    if (n < 0) {
        std::fprintf(stderr, "tokenize failed\n");
        std::exit(1);
    }
    tokens.resize(static_cast<std::size_t>(n));
    return tokens;
}

void print_ids(const std::vector<llama_token>& ids) {
    for (std::size_t i = 0; i < ids.size(); ++i) std::printf(i ? " %d" : "%d", ids[i]);
    std::printf("\n");
}

void quiet_log(ggml_log_level level, const char* text, void*) {
    if (level >= GGML_LOG_LEVEL_ERROR) std::fputs(text, stderr);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (!((mode == "tokenize" && argc == 4) || (mode == "eval" && argc == 6))) {
        std::fprintf(stderr, "usage: llama_ref tokenize <model> <strings.bin>\n"
                             "       llama_ref eval <model> <prompt.txt> <n_generate> <logits.bin>\n");
        return 2;
    }

    llama_log_set(quiet_log, nullptr);
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    model_params.vocab_only = mode == "tokenize";
    llama_model* model = llama_model_load_from_file(argv[2], model_params);
    if (model == nullptr) {
        std::fprintf(stderr, "cannot load %s\n", argv[2]);
        return 1;
    }
    const llama_vocab* vocab = llama_model_get_vocab(model);

    if (mode == "tokenize") {
        const std::string data = read_file(argv[3]);
        std::size_t pos = 0;
        while (pos + 4 <= data.size()) {
            std::uint32_t len = 0;
            std::copy_n(data.data() + pos, 4, reinterpret_cast<char*>(&len));
            pos += 4;
            if (len > data.size() - pos) {
                std::fprintf(stderr, "truncated strings file\n");
                return 1;
            }
            print_ids(tokenize(vocab, data.substr(pos, len)));
            pos += len;
        }
        llama_model_free(model);
        return 0;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = kContext;
    ctx_params.n_batch = kContext;
    ctx_params.n_threads = kThreads;
    ctx_params.n_threads_batch = kThreads;
    ctx_params.type_k = GGML_TYPE_F32;  // keep the KV cache in full precision for a tight comparison
    ctx_params.type_v = GGML_TYPE_F32;
    ctx_params.offload_kqv = false;
    llama_context* ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "cannot create context\n");
        return 1;
    }

    std::vector<llama_token> prompt = tokenize(vocab, read_file(argv[3]));
    const int n_generate = std::atoi(argv[4]);
    if (prompt.empty() || prompt.size() + static_cast<std::size_t>(std::max(n_generate, 0)) > kContext) {
        std::fprintf(stderr, "prompt is empty or does not fit the context\n");
        return 1;
    }
    print_ids(prompt);

    if (llama_decode(ctx, llama_batch_get_one(prompt.data(), static_cast<int32_t>(prompt.size()))) != 0) {
        std::fprintf(stderr, "decode failed\n");
        return 1;
    }
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const float* logits = llama_get_logits_ith(ctx, -1);
    std::ofstream(argv[5], std::ios::binary)
        .write(reinterpret_cast<const char*>(logits), static_cast<std::streamsize>(n_vocab) * 4);

    std::vector<llama_token> generated;
    for (int i = 0; i < n_generate; ++i) {
        logits = llama_get_logits_ith(ctx, -1);
        llama_token next = static_cast<llama_token>(std::max_element(logits, logits + n_vocab) - logits);
        generated.push_back(next);
        if (llama_decode(ctx, llama_batch_get_one(&next, 1)) != 0) {
            std::fprintf(stderr, "decode failed\n");
            return 1;
        }
    }
    print_ids(generated);

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
