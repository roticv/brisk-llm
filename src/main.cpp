#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <filesystem>
#include <thread>
#include <type_traits>
#include <vector>

#include "gguf.h"
#include "kernels.h"
#include "mapped_file.h"
#include "model.h"
#if BRISK_METAL
#include "metal/metal_session.h"
#endif
#include "catalog.h"
#include "chat.h"
#include "memory.h"
#include "server.h"
#include "sampler.h"
#include "speculative.h"
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
    bool gpu = false;              // run the model on the Metal GPU
    bool warmup = false;           // evaluate the prompt once untimed before the timed run
    std::string draft;             // "lookup", or a path to a smaller model with the same tokenizer
    std::size_t draft_tokens = 5;  // guesses per round
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

    std::string text = options.prompt;
    if (options.chat) {
        brisk::ChatOptions chat_options;
        chat_options.think = options.think;
        text = brisk::render_chat(brisk::chat_format_of(file), {{"user", options.prompt}}, chat_options);
    }
    std::vector<brisk::Token> prompt = tokenizer.encode(text);
    if (tokenizer.bos() >= 0) prompt.insert(prompt.begin(), tokenizer.bos());
    if (prompt.empty()) throw std::runtime_error("the prompt is empty");
    const brisk::ModelConfig config = brisk::Model::parse_config(file);
    if (prompt.size() + options.max_tokens > config.context_length) {
        throw std::runtime_error("prompt plus generated tokens exceed the model's context length");
    }
    if (options.print_ids) print_ids(prompt);

    // Room for the prompt, the answer, and a round of speculative guesses.
    const std::size_t max_context = prompt.size() + options.max_tokens + options.draft_tokens + 1;
    brisk::check_memory(brisk::estimate_memory(file, config, /*repack=*/!options.gpu, max_context), max_context, "generate");
    const brisk::Model model(file, /*repack=*/!options.gpu);
    const double load_seconds = seconds_since(load_start);
    std::unique_ptr<brisk::Engine> engine;
    if (options.gpu) {
#if BRISK_METAL
        engine = std::make_unique<brisk::MetalSession>(model, mapped.bytes(), max_context);
#else
        throw std::runtime_error("this build has no GPU backend");
#endif
    } else {
        engine = std::make_unique<brisk::Session>(model, options.threads, max_context);
    }
    brisk::Engine& session = *engine;
    if (options.warmup) {
        // Like llama-bench: an untimed pass first, so the timed one sees warmed
        // clocks and resident weights.
        session.eval(prompt);
        session.rollback(0);
    }
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

    if (!options.draft.empty()) {
        // Speculative decoding reproduces greedy output exactly, so it only applies there.
        if (options.sampler.temperature > 0.0f) throw std::runtime_error("--draft requires greedy sampling (--temp 0)");
        std::unique_ptr<brisk::MappedFile> draft_mapped;
        std::unique_ptr<brisk::GgufFile> draft_file;
        std::unique_ptr<brisk::Model> draft_model;
        std::unique_ptr<brisk::Drafter> drafter;
        if (options.draft == "lookup") {
            drafter = std::make_unique<brisk::LookupDrafter>();
        } else {
            draft_mapped = std::make_unique<brisk::MappedFile>(options.draft);
            draft_file = std::make_unique<brisk::GgufFile>(brisk::GgufFile::parse(draft_mapped->bytes()));
            draft_model = std::make_unique<brisk::Model>(*draft_file);
            if (draft_model->config().vocab_size != model.config().vocab_size) {
                throw std::runtime_error("the draft model's vocabulary differs from the target's");
            }
            drafter = std::make_unique<brisk::ModelDrafter>(*draft_model, options.threads, max_context);
        }
        drafter->observe(prompt);
        // The target has evaluated the whole prompt; step back one so the last
        // token can be re-evaluated together with the guesses.
        session.rollback(prompt.size() - 1);
        brisk::Speculator speculator(session, *drafter, prompt.back(), options.draft_tokens);

        bool done = false;
        while (!done && generated.size() < options.max_tokens) {
            for (const brisk::Token next : speculator.step()) {
                if ((options.stop_at_end && tokenizer.is_end_of_generation(next)) || generated.size() >= options.max_tokens) {
                    done = true;
                    break;
                }
                generated.push_back(next);
                if (!options.print_ids && !tokenizer.is_control(next)) {
                    const std::string_view piece = tokenizer.piece(next);
                    std::fwrite(piece.data(), 1, piece.size(), stdout);
                    std::fflush(stdout);
                }
            }
        }
        const double generate_seconds = seconds_since(generate_start);
        if (options.print_ids) print_ids(generated);
        else std::printf("\n");
        std::fflush(stdout);
        const brisk::SpeculativeStats& st = speculator.stats();
        std::fprintf(stderr,
                     "load %.2f s | prompt %zu tokens, %.1f tokens/s | generation %zu tokens, %.1f tokens/s | "
                     "%zu drafted rounds + %zu plain, %zu of %zu guesses accepted | %zu threads, %.*s kernels\n",
                     load_seconds, prompt.size(), static_cast<double>(prompt.size()) / prompt_seconds, generated.size(),
                     static_cast<double>(generated.size()) / generate_seconds, st.rounds, st.plain, st.accepted, st.drafted,
                     options.threads, static_cast<int>(brisk::kernels::kernel_set_name().size()),
                     brisk::kernels::kernel_set_name().data());
        return 0;
    }

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

    if (const auto* cpu = dynamic_cast<brisk::Session*>(engine.get()); cpu != nullptr && std::getenv("BRISK_PROFILE") != nullptr) {
        const brisk::Session::Profile& p = cpu->profile();
        std::fprintf(stderr, "profile: matmul %.2f s | quantize %.2f s | attention %.2f s | norm+rope %.2f s | activation %.2f s\n",
                     p.matmul - p.quantize, p.quantize, p.attention, p.norm_rope, p.activation);
    }

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
int cmd_perplexity(const std::string& model_path, const std::string& text, std::size_t threads, bool gpu) {
    const brisk::MappedFile mapped(model_path);
    const brisk::GgufFile file = brisk::GgufFile::parse(mapped.bytes());
    const brisk::Tokenizer tokenizer(file);
    const std::vector<brisk::Token> tokens = tokenizer.encode(text);
    if (tokens.size() < 2) throw std::runtime_error("the text is too short to score");
    const brisk::ModelConfig config = brisk::Model::parse_config(file);
    if (tokens.size() > config.context_length) throw std::runtime_error("the text exceeds the context length");
    brisk::check_memory(brisk::estimate_memory(file, config, /*repack=*/!gpu, tokens.size()), tokens.size(), "perplexity");
    const brisk::Model model(file, /*repack=*/!gpu);

    std::unique_ptr<brisk::Engine> engine;
    if (gpu) {
#if BRISK_METAL
        engine = std::make_unique<brisk::MetalSession>(model, mapped.bytes(), tokens.size());
#else
        throw std::runtime_error("this build has no GPU backend");
#endif
    } else {
        engine = std::make_unique<brisk::Session>(model, threads, tokens.size());
    }
    const auto start = std::chrono::steady_clock::now();
    const double log_likelihood = engine->log_likelihood(tokens);
    const double scored = static_cast<double>(tokens.size() - 1);
    std::printf("tokens %zu | nll %.4f | perplexity %.4f | %.1f s\n", tokens.size(), -log_likelihood / scored,
                std::exp(-log_likelihood / scored), seconds_since(start));
    return 0;
}

int usage() {
    std::fprintf(stderr,
                 "usage: brisk info [--tensors | --memory [<context>]] <model.gguf>   (--memory: what a session needs)\n"
                 "       brisk tokenize <model.gguf> [--no-special] (--batch <strings.bin> | <text>)\n"
                 "       brisk generate <model.gguf> (-p <prompt> | -f <prompt-file>) [options]\n"
                 "       brisk perplexity <model.gguf> -f <text-file> [-t <threads>] [--gpu]\n"
                 "       brisk serve <model.gguf> [--host <addr>] [--port <n>] [-t <threads>] [--gpu] [--ctx <tokens>]\n"
                 "                  [--name <id>] [-v] [--no-prefetch] [--mlock]\n"
                 "                  OpenAI-compatible API (/v1/chat/completions, /v1/models); the weights are paged in at start\n"
                 "       brisk template <model.gguf> (<role> <content>)...   the rendered chat prompt\n"
                 "       brisk pull [<name>...] [--dir <dir>]   download a verified model file (no name: list them)\n"
                 "\n"
                 "generate options:\n"
                 "  -n <count>            most tokens to generate (default 128)\n"
                 "  -t <threads>          CPU threads to use (default: all cores)\n"
                 "  --gpu                 run on the Metal GPU (Apple silicon; Q4_0 models)\n"
                 "  --warmup              evaluate the prompt once untimed first (benchmarking)\n"
                 "  --chat                treat the prompt as a user message\n"
                 "  --no-think            with --chat, answer without a reasoning block\n"
                 "  --temp <t>            sampling temperature; 0 always picks the likeliest token (default 0)\n"
                 "  --top-k <k>           sample from the k likeliest tokens; 0 = all (default 20)\n"
                 "  --top-p <p>           sample from the smallest set reaching probability p (default 0.95)\n"
                 "  --seed <n>            random seed (default 0)\n"
                 "  --ids                 print token ids (prompt line, then generated line) instead of text\n"
                 "  --no-stop             keep generating past the end-of-turn token\n"
                 "  --dump-logits <file>  write the float32 logits that follow the prompt\n"
                 "  --draft <model|lookup> speculative decoding: guess with a smaller model, or by\n"
                 "                        looking up repeats in the text (greedy sampling only)\n"
                 "  --draft-tokens <k>    guesses per round (default 5)\n");
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
        const bool memory = args.size() > 1 && args[1] == "--memory";
        if (memory) {
            // brisk info --memory <model> [<context>]: what a session needs, CPU and GPU, against what is free.
            if (args.size() < 3 || args.size() > 4) return usage();
            const brisk::MappedFile mapped(args[2]);
            const brisk::GgufFile file = brisk::GgufFile::parse(mapped.bytes());
            const brisk::ModelConfig config = brisk::Model::parse_config(file);
            const std::size_t context = args.size() == 4 ? std::stoul(args[3]) : std::min<std::size_t>(config.context_length, 8192);
            std::printf("context %zu tokens\nCPU: %s\nGPU: %s\navailable now: %.2f GB\n", context,
                        brisk::estimate_memory(file, config, true, context).describe().c_str(),
                        brisk::estimate_memory(file, config, false, context).describe().c_str(),
                        static_cast<double>(brisk::available_memory()) / (1024.0 * 1024.0 * 1024.0));
            return 0;
        }
        if (args.size() != (list_tensors ? 3u : 2u)) return usage();
        return cmd_info(args.back(), list_tensors);
    }

    if (command == "pull") {
        // brisk pull [<name>...] [--dir <dir>]: downloads verified files; no name lists them.
        std::string dir = "models";
        std::vector<std::string> names;
        for (std::size_t i = 1; i < args.size(); ++i) {
            if (args[i] == "--dir" && i + 1 < args.size()) dir = args[++i];
            else names.push_back(args[i]);
        }
        if (names.empty()) {
            std::printf("%-18s %9s  %-40s %s\n", "name", "size", "file", "notes");
            for (const brisk::CatalogEntry& e : brisk::catalog()) {
                const bool present = std::filesystem::exists(std::filesystem::path(dir) / std::string(e.file));
                std::printf("%-18.*s %7.2f GB  %-40.*s %.*s%s\n", static_cast<int>(e.name.size()), e.name.data(),
                            static_cast<double>(e.bytes) / (1024.0 * 1024.0 * 1024.0), static_cast<int>(e.file.size()), e.file.data(),
                            static_cast<int>(e.note.size()), e.note.data(), present ? " [present]" : "");
            }
            std::printf("\nbrisk pull <name> downloads into %s/ (resumable); then: brisk generate %s/<file> --chat -p \"...\"\n", dir.c_str(), dir.c_str());
            return 0;
        }
        for (const std::string& name : names) {
            const brisk::CatalogEntry* e = brisk::catalog_find(name);
            if (e == nullptr) throw std::runtime_error("unknown model '" + name + "'; brisk pull lists the names");
            std::fprintf(stderr, "pulling %.*s (%.2f GB)\n", static_cast<int>(e->file.size()), e->file.data(),
                         static_cast<double>(e->bytes) / (1024.0 * 1024.0 * 1024.0));
            const std::string path = brisk::pull(*e, dir, false);
            std::printf("%s\n", path.c_str());
        }
        return 0;
    }

    if (command == "serve") {
        if (args.size() < 2) return usage();
        brisk::ServeOptions o;
        o.model_path = args[1];
        o.threads = static_cast<std::size_t>(std::thread::hardware_concurrency());
        for (std::size_t i = 2; i < args.size(); ++i) {
            const std::string& flag = args[i];
            auto value = [&]() -> const std::string& {
                if (i + 1 >= args.size()) throw std::runtime_error(flag + " needs a value");
                return args[++i];
            };
            if (flag == "--host") o.host = value();
            else if (flag == "--port") o.port = std::stoi(value());
            else if (flag == "-t") o.threads = std::stoul(value());
            else if (flag == "--gpu") o.gpu = true;
            else if (flag == "--ctx") o.max_context = std::stoul(value());
            else if (flag == "--name") o.name = value();
            else if (flag == "-v" || flag == "--verbose") o.verbose = true;
            else if (flag == "--no-prefetch") o.prefetch = false;
            else if (flag == "--mlock") o.lock = true;
            else return usage();
        }
        return brisk::serve(o);
    }

    if (command == "template") {
        // brisk template <model> (<role> <content>)...: the rendered chat prompt, for checks against llama.cpp.
        if (args.size() < 4 || args.size() % 2 != 0) return usage();
        const brisk::MappedFile mapped(args[1]);
        const brisk::GgufFile file = brisk::GgufFile::parse(mapped.bytes());
        std::vector<brisk::ChatMessage> messages;
        for (std::size_t i = 2; i + 1 < args.size(); i += 2) messages.push_back({args[i], args[i + 1]});
        const std::string text = brisk::render_chat(brisk::chat_format_of(file), messages);
        std::fwrite(text.data(), 1, text.size(), stdout);
        return 0;
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
        bool gpu = false;
        for (std::size_t i = 4; i < args.size(); ++i) {
            if (args[i] == "-t" && i + 1 < args.size()) threads = parse_number<std::size_t>("-t", args[++i]);
            else if (args[i] == "--gpu") gpu = true;
            else return usage();
        }
        return cmd_perplexity(args[1], read_text_file(args[3]), threads, gpu);
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
            else if (flag == "--gpu") options.gpu = true;
            else if (flag == "--warmup") options.warmup = true;
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
            else if (flag == "--draft") options.draft = args[++i];
            else if (flag == "--draft-tokens") options.draft_tokens = parse_number<std::size_t>(flag, args[++i]);
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
