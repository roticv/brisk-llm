#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <vector>

#include "gguf.h"
#include "kernels.h"
#include "mapped_file.h"
#include "model.h"
#include "sampler.h"
#include "tokenizer.h"

namespace {

using brisk::GgufArray;
using brisk::GgufType;
using brisk::GgufValue;

constexpr std::size_t kArrayPreview = 4;
constexpr std::size_t kStringPreview = 80;

void print_string(std::string_view s) {
    const std::string_view shown = s.substr(0, kStringPreview);
    std::printf("\"");
    for (const char c : shown) {
        if (c == '\n') std::printf("\\n");
        else std::putchar(c);
    }
    std::printf("\"%s", s.size() > shown.size() ? "..." : "");
}

void print_array(const GgufArray& array) {
    std::printf("[%s x %llu]", std::string(to_string(array.elem_type)).c_str(),
                static_cast<unsigned long long>(array.count));
    if (array.elem_type != GgufType::String) return;
    const auto items = array.strings();
    for (std::size_t i = 0; i < items.size() && i < kArrayPreview; ++i) {
        std::printf(" ");
        print_string(items[i]);
    }
    if (items.size() > kArrayPreview) std::printf(" ...");
}

struct ValuePrinter {
    void operator()(std::string_view s) const { print_string(s); }
    void operator()(const GgufArray& a) const { print_array(a); }
    void operator()(bool b) const { std::printf("%s", b ? "true" : "false"); }
    void operator()(float f) const { std::printf("%g", static_cast<double>(f)); }
    void operator()(double d) const { std::printf("%g", d); }
    void operator()(std::uint64_t u) const { std::printf("%llu", static_cast<unsigned long long>(u)); }
    void operator()(std::int64_t i) const { std::printf("%lld", static_cast<long long>(i)); }
    void operator()(std::uint8_t u) const { (*this)(static_cast<std::uint64_t>(u)); }
    void operator()(std::uint16_t u) const { (*this)(static_cast<std::uint64_t>(u)); }
    void operator()(std::uint32_t u) const { (*this)(static_cast<std::uint64_t>(u)); }
    void operator()(std::int8_t i) const { (*this)(static_cast<std::int64_t>(i)); }
    void operator()(std::int16_t i) const { (*this)(static_cast<std::int64_t>(i)); }
    void operator()(std::int32_t i) const { (*this)(static_cast<std::int64_t>(i)); }
};

int cmd_info(const std::string& path, bool list_tensors) {
    const brisk::MappedFile mapped(path);
    const brisk::GgufFile file = brisk::GgufFile::parse(mapped.bytes());

    std::printf("GGUF v%u, %zu metadata entries, %zu tensors\n\n", file.version(), file.metadata().size(),
                file.tensors().size());
    for (const auto& [key, value] : file.metadata()) {
        std::printf("%-44.*s ", static_cast<int>(key.size()), key.data());
        std::visit(ValuePrinter{}, value);
        std::printf("\n");
    }

    struct Total {
        std::size_t tensors = 0;
        std::uint64_t elements = 0;
        std::uint64_t bytes = 0;
    };
    std::map<std::string_view, Total> totals;
    std::printf("\n");
    for (const brisk::TensorInfo& t : file.tensors()) {
        Total& total = totals[to_string(t.type)];
        total.tensors += 1;
        total.elements += t.n_elements();
        total.bytes += t.data.size();
        if (!list_tensors) continue;

        std::string shape;
        for (const std::uint64_t d : t.dims) shape += (shape.empty() ? "" : " x ") + std::to_string(d);
        std::printf("%-44.*s %-6s %-18s %10.2f MB\n", static_cast<int>(t.name.size()), t.name.data(),
                    std::string(to_string(t.type)).c_str(), shape.c_str(), static_cast<double>(t.data.size()) / 1e6);
    }
    if (list_tensors) std::printf("\n");

    for (const auto& [type, total] : totals) {
        std::printf("%-6s %4zu tensors %8.1f M weights %9.1f MB\n", std::string(type).c_str(), total.tensors,
                    static_cast<double>(total.elements) / 1e6, static_cast<double>(total.bytes) / 1e6);
    }
    return 0;
}

void print_ids(const std::vector<brisk::Token>& ids) {
    for (std::size_t i = 0; i < ids.size(); ++i) std::printf(i ? " %d" : "%d", ids[i]);
    std::printf("\n");
}

// `batch_path` names a file of [u32 length][bytes] records, one line of ids
// printed per record; it exists so tests can cover text with any bytes in it.
int cmd_tokenize(const std::string& model_path, bool parse_special, const std::string& batch_path,
                 const std::string& text) {
    const brisk::MappedFile mapped(model_path);
    const brisk::GgufFile file = brisk::GgufFile::parse(mapped.bytes());
    const brisk::Tokenizer tokenizer(file);

    if (batch_path.empty()) {
        print_ids(tokenizer.encode(text, parse_special));
        return 0;
    }

    const brisk::MappedFile batch(batch_path);
    const auto bytes = batch.bytes();
    std::size_t pos = 0;
    while (pos + 4 <= bytes.size()) {
        std::uint32_t len = 0;
        std::memcpy(&len, bytes.data() + pos, 4);
        pos += 4;
        if (len > bytes.size() - pos) throw std::runtime_error("truncated strings file");
        print_ids(tokenizer.encode({reinterpret_cast<const char*>(bytes.data()) + pos, len}, parse_special));
        pos += len;
    }
    return 0;
}

struct GenerateOptions {
    std::string model_path;
    std::string prompt;
    std::size_t max_tokens = 128;
    bool chat = false;         // wrap the prompt as a single user turn
    bool think = true;         // with chat: let the model reason before answering
    bool print_ids = false;    // print token ids instead of text
    bool stop_at_end = true;   // stop when the model ends its turn
    std::string logits_path;   // if set, the logits after the prompt are written here
    std::size_t threads = std::thread::hardware_concurrency();
    brisk::SamplerConfig sampler;
};

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

int cmd_generate(const GenerateOptions& options) {
    const auto load_start = std::chrono::steady_clock::now();
    const brisk::MappedFile mapped(options.model_path);
    const brisk::GgufFile file = brisk::GgufFile::parse(mapped.bytes());
    const brisk::Tokenizer tokenizer(file);
    const brisk::Model model(file);
    const double load_seconds = seconds_since(load_start);

    std::string text = options.prompt;
    if (options.chat) {
        // Qwen's ChatML turn format. An empty think block tells Qwen3 to answer directly.
        text = "<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n";
        if (!options.think) text += "<think>\n\n</think>\n\n";
    }
    const std::vector<brisk::Token> prompt = tokenizer.encode(text);
    if (prompt.empty()) throw std::runtime_error("the prompt is empty");
    if (prompt.size() + options.max_tokens > model.config().context_length) {
        throw std::runtime_error("prompt plus generated tokens exceed the model's context length");
    }
    if (options.print_ids) print_ids(prompt);

    brisk::Session session(model, options.threads);
    const auto prompt_start = std::chrono::steady_clock::now();
    std::span<const float> logits = session.eval(prompt);
    const double prompt_seconds = seconds_since(prompt_start);

    if (!options.logits_path.empty()) {
        std::ofstream out(options.logits_path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(logits.data()), static_cast<std::streamsize>(logits.size_bytes()));
        if (!out) throw std::runtime_error("cannot write '" + options.logits_path + "'");
    }

    brisk::Sampler sampler(options.sampler);
    std::vector<brisk::Token> generated;
    const auto generate_start = std::chrono::steady_clock::now();
    while (generated.size() < options.max_tokens) {
        const brisk::Token next = sampler.sample(logits);
        if (options.stop_at_end && tokenizer.is_end_of_generation(next)) break;
        generated.push_back(next);
        if (!options.print_ids && !tokenizer.is_control(next)) {
            const std::string_view piece = tokenizer.piece(next);
            std::fwrite(piece.data(), 1, piece.size(), stdout);
            std::fflush(stdout);
        }
        // The logits after the final token are never used, so skip computing them.
        if (generated.size() < options.max_tokens) logits = session.eval(next);
    }
    const double generate_seconds = seconds_since(generate_start);

    if (options.print_ids) print_ids(generated);
    else std::printf("\n");
    std::fflush(stdout);

    // The last generated token is not evaluated, so one fewer pass was timed.
    const std::size_t timed = generated.empty() ? 0 : generated.size() - (generated.size() == options.max_tokens);
    std::fprintf(stderr,
                 "load %.2f s | prompt %zu tokens, %.1f tokens/s | generation %zu tokens, %.1f tokens/s | %zu threads, %.*s kernels\n",
                 load_seconds, prompt.size(), static_cast<double>(prompt.size()) / prompt_seconds, generated.size(),
                 timed > 0 ? static_cast<double>(timed) / generate_seconds : 0.0, options.threads,
                 static_cast<int>(brisk::kernels::kernel_set_name().size()), brisk::kernels::kernel_set_name().data());
    return 0;
}

// Scores a text: the model's perplexity over it, the standard check that an
// implementation is right, since kernel errors raise it and rounding noise averages out.
int cmd_perplexity(const std::string& model_path, const std::string& text, std::size_t threads) {
    const brisk::MappedFile mapped(model_path);
    const brisk::GgufFile file = brisk::GgufFile::parse(mapped.bytes());
    const brisk::Tokenizer tokenizer(file);
    const brisk::Model model(file);
    const std::vector<brisk::Token> tokens = tokenizer.encode(text);
    if (tokens.size() < 2) throw std::runtime_error("the text is too short to score");
    if (tokens.size() > model.config().context_length) throw std::runtime_error("the text exceeds the context length");

    brisk::Session session(model, threads);
    const auto start = std::chrono::steady_clock::now();
    const double log_likelihood = session.log_likelihood(tokens);
    const double scored = static_cast<double>(tokens.size() - 1);
    std::printf("tokens %zu | nll %.4f | perplexity %.4f | %.1f s\n", tokens.size(), -log_likelihood / scored,
                std::exp(-log_likelihood / scored), seconds_since(start));
    return 0;
}

int usage() {
    std::fprintf(stderr,
                 "usage: brisk info [--tensors] <model.gguf>\n"
                 "       brisk tokenize <model.gguf> [--no-special] (--batch <strings.bin> | <text>)\n"
                 "       brisk generate <model.gguf> (-p <prompt> | -f <prompt-file>) [options]\n"
                 "       brisk perplexity <model.gguf> -f <text-file> [-t <threads>]\n"
                 "\n"
                 "generate options:\n"
                 "  -n <count>            most tokens to generate (default 128)\n"
                 "  -t <threads>          CPU threads to use (default: all cores)\n"
                 "  --chat                treat the prompt as a user message\n"
                 "  --no-think            with --chat, answer without a reasoning block\n"
                 "  --temp <t>            sampling temperature; 0 always picks the likeliest token (default 0)\n"
                 "  --top-k <k>           sample from the k likeliest tokens; 0 = all (default 20)\n"
                 "  --top-p <p>           sample from the smallest set reaching probability p (default 0.95)\n"
                 "  --seed <n>            random seed (default 0)\n"
                 "  --ids                 print token ids (prompt line, then generated line) instead of text\n"
                 "  --no-stop             keep generating past the end-of-turn token\n"
                 "  --dump-logits <file>  write the float32 logits that follow the prompt\n");
    return 2;
}

std::string read_text_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open '" + path + "'");
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

template <typename T>
T parse_number(const std::string& flag, const std::string& value) {
    try {
        std::size_t used = 0;
        T out{};
        if constexpr (std::is_floating_point_v<T>) {
            out = static_cast<T>(std::stod(value, &used));
        } else {
            if (!value.empty() && value[0] == '-') throw std::invalid_argument("negative");
            const unsigned long long parsed = std::stoull(value, &used);
            out = static_cast<T>(parsed);
            if (static_cast<unsigned long long>(out) != parsed) throw std::out_of_range("too large");
        }
        if (used != value.size()) throw std::invalid_argument("trailing characters");
        return out;
    } catch (const std::logic_error&) {
        throw std::runtime_error("invalid value '" + value + "' for " + flag);
    }
}

int run(const std::vector<std::string>& args) {
    if (args.empty()) return usage();
    const std::string& command = args[0];

    if (command == "info") {
        const bool list_tensors = args.size() > 1 && args[1] == "--tensors";
        if (args.size() != (list_tensors ? 3u : 2u)) return usage();
        return cmd_info(args.back(), list_tensors);
    }

    if (command == "tokenize") {
        if (args.size() < 3) return usage();
        bool parse_special = true;
        std::string batch_path;
        std::string text;
        bool have_input = false;
        for (std::size_t i = 2; i < args.size(); ++i) {
            if (args[i] == "--no-special") {
                parse_special = false;
            } else if (args[i] == "--batch" && i + 1 < args.size() && !have_input) {
                batch_path = args[++i];
                have_input = true;
            } else if (!have_input) {
                text = args[i];
                have_input = true;
            } else {
                return usage();
            }
        }
        if (!have_input) return usage();
        return cmd_tokenize(args[1], parse_special, batch_path, text);
    }

    if (command == "perplexity") {
        if (args.size() < 4 || args[2] != "-f") return usage();
        std::size_t threads = std::thread::hardware_concurrency();
        if (args.size() == 6 && args[4] == "-t") threads = parse_number<std::size_t>("-t", args[5]);
        else if (args.size() != 4) return usage();
        return cmd_perplexity(args[1], read_text_file(args[3]), threads);
    }

    if (command == "generate") {
        if (args.size() < 2) return usage();
        GenerateOptions options;
        options.model_path = args[1];
        bool have_prompt = false;
        for (std::size_t i = 2; i < args.size(); ++i) {
            const std::string& flag = args[i];
            const bool has_value = i + 1 < args.size();
            if (flag == "--chat") options.chat = true;
            else if (flag == "--no-think") options.think = false;
            else if (flag == "--ids") options.print_ids = true;
            else if (flag == "--no-stop") options.stop_at_end = false;
            else if (!has_value) return usage();
            else if (flag == "-p") { options.prompt = args[++i]; have_prompt = true; }
            else if (flag == "-f") { options.prompt = read_text_file(args[++i]); have_prompt = true; }
            else if (flag == "-n") options.max_tokens = parse_number<std::size_t>(flag, args[++i]);
            else if (flag == "-t") options.threads = parse_number<std::size_t>(flag, args[++i]);
            else if (flag == "--temp") options.sampler.temperature = parse_number<float>(flag, args[++i]);
            else if (flag == "--top-k") options.sampler.top_k = parse_number<std::uint32_t>(flag, args[++i]);
            else if (flag == "--top-p") options.sampler.top_p = parse_number<float>(flag, args[++i]);
            else if (flag == "--seed") options.sampler.seed = parse_number<std::uint64_t>(flag, args[++i]);
            else if (flag == "--dump-logits") options.logits_path = args[++i];
            else return usage();
        }
        if (!have_prompt) return usage();
        return cmd_generate(options);
    }

    return usage();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run({argv + 1, argv + argc});
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
