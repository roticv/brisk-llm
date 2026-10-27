#include "server.h"

#include <algorithm>
#include <cctype>
#include <atomic>
#include <chrono>
#include <csignal>
#include <thread>

#include <unistd.h>
#include <cstdio>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "chat.h"
#include "engine.h"
#include "gguf.h"
#include "mapped_file.h"
#include "memory.h"
#include "model.h"
#include "sampler.h"
#include "tokenizer.h"
#if BRISK_METAL
#include "metal/metal_session.h"
#endif

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wimplicit-int-conversion"
#pragma clang diagnostic ignored "-Wshorten-64-to-32"
#include "httplib.h"
#include "json.hpp"
#pragma clang diagnostic pop

namespace brisk {

using json = nlohmann::ordered_json;  // ordered: tool JSON is re-rendered into prompts as given

namespace {

// Set by SIGINT/SIGTERM: the generation in flight ends at its next token, the
// listener closes, and the process exits once the request has been answered.
std::atomic<bool> g_stopping{false};

void on_signal(int) {
    g_stopping.store(true);
    const char msg[] = "brisk serve: stopping after the request in flight\n";
    (void)!::write(2, msg, sizeof(msg) - 1);
}

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

std::string basename_of(const std::string& path) {
    const std::size_t slash = path.find_last_of('/');
    std::string name = slash == std::string::npos ? path : path.substr(slash + 1);
    if (name.size() > 5 && name.compare(name.size() - 5, 5, ".gguf") == 0) name.resize(name.size() - 5);
    return name;
}

// Text that is safe to emit now: a trailing incomplete UTF-8 sequence (a
// byte-fallback token can end mid-character) is held back for the next piece.
std::size_t complete_utf8_prefix(const std::string& s) {
    std::size_t i = s.size();
    // Walk back over continuation bytes to the last lead byte.
    std::size_t back = 0;
    while (i > 0 && back < 4 && (static_cast<unsigned char>(s[i - 1]) & 0xC0) == 0x80) {
        --i;
        ++back;
    }
    if (i == 0) return s.size();  // only continuation bytes: nothing sensible to hold
    const unsigned char lead = static_cast<unsigned char>(s[i - 1]);
    std::size_t need = 0;
    if ((lead & 0x80) == 0) need = 1;
    else if ((lead & 0xE0) == 0xC0) need = 2;
    else if ((lead & 0xF0) == 0xE0) need = 3;
    else if ((lead & 0xF8) == 0xF0) need = 4;
    else return s.size();  // invalid lead byte: emit as is
    const std::size_t have = 1 + back;
    return have >= need ? s.size() : i - 1;  // hold the incomplete sequence back
}

// What a request asks for, after the OpenAI fields are read.
std::string_view trim(std::string_view s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.remove_prefix(1);
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.remove_suffix(1);
    return s;
}

}  // namespace

std::string render_chat_request(const GgufFile& file, const json& body, std::vector<ChatMessage>& messages, ChatOptions& co) {
    if (!body.contains("messages") || !body["messages"].is_array()) throw std::invalid_argument("'messages' must be an array");
        for (const json& m : body["messages"]) {
            ChatMessage msg;
            msg.role = m.value("role", "");
            msg.tool_call_id = m.value("tool_call_id", "");
            if (m.contains("reasoning_content") && m["reasoning_content"].is_string()) msg.reasoning = m["reasoning_content"];
            if (m.contains("tool_calls") && m["tool_calls"].is_array()) {
                for (const json& c : m["tool_calls"]) {
                    const json& f = c.contains("function") ? c["function"] : c;
                    const json& args = f.contains("arguments") ? f["arguments"] : json::object();
                    msg.tool_calls.push_back({f.value("name", ""), args.is_string() ? args.get<std::string>() : args.dump(), c.value("id", "")});
                }
            }
            const json& content = m.contains("content") ? m["content"] : json();
            if (content.is_string()) {
                msg.content = content.get<std::string>();
            } else if (content.is_array()) {  // content parts: the text ones are joined
                for (const json& part : content) {
                    if (part.value("type", "") == "text") msg.content += part.value("text", "");
                }
            } else if (!content.is_null()) {
                throw std::invalid_argument("message content must be a string or an array of parts");
            }
            messages.push_back(std::move(msg));
        }
        if (body.contains("tools") && body["tools"].is_array()) {
            // Normalised as llama.cpp does before rendering: type, then name, description and
            // parameters (an empty object when missing); other keys dropped.
            for (const json& t : body["tools"]) {
                const json& f = t.contains("function") ? t["function"] : t;
                if (!f.is_object() || !f.contains("name")) throw std::invalid_argument("each tool needs a function name");
                json tool = {{"type", "function"},
                             {"function", {{"name", f["name"]},
                                           {"description", f.value("description", "")},
                                           {"parameters", f.contains("parameters") ? f["parameters"] : json::object()}}}};
                co.tools.push_back(tool.dump());
            }
        }
        // Qwen3's reasoning: enable_thinking at the top level or in chat_template_kwargs (both are in use).
        if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object()) {
            co.think = body["chat_template_kwargs"].value("enable_thinking", true);
        }
        co.think = body.value("enable_thinking", co.think);
        if (body.contains("reasoning") && body["reasoning"].is_object()) {
            co.think = body["reasoning"].value("effort", "medium") != "none";
        }
        // Gemma 4 thinks only when asked (its template's default), by any of the three forms.
        co.think_requested = body.value("enable_thinking", false) ||
                             (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object() &&
                              body["chat_template_kwargs"].value("enable_thinking", false)) ||
                             (body.contains("reasoning") && body["reasoning"].is_object() &&
                              body["reasoning"].value("effort", "none") != "none");
        return render_chat(file, messages, co);
}

namespace {

struct Request {
    std::vector<Token> prompt;
    std::size_t max_tokens = 0;
    SamplerConfig sampler;
    std::vector<std::string> stop;
    bool stream = false;
    bool echo_prompt_tokens = false;
    bool separate_reasoning = true;  // a <think> block goes to reasoning_content rather than inline in content
    bool tools = false;              // the request offered tools: <tool_call> blocks (Qwen) are parsed into tool_calls
    std::vector<std::string> tool_names;  // a call is only a call when it names one of these
    bool hold_content = false;       // Llama 3 with tools: the whole answer may be a call, so content waits for the end
};

class Server {
public:
    // The context a session holds: the request, or the model's, capped at 8192.
    static std::size_t context_for(const ServeOptions& options, const ModelConfig& c) {
        return options.max_context != 0 ? std::min<std::size_t>(options.max_context, c.context_length)
                                        : std::min<std::size_t>(c.context_length, 8192);
    }
    // Refuses before the weights load when the model would not fit (the check throws).
    static const GgufFile& checked(const GgufFile& file, const ServeOptions& options) {
        const ModelConfig c = Model::parse_config(file);
        const std::size_t context = context_for(options, c);
        check_memory(estimate_memory(file, c, /*repack=*/!options.gpu, context), context, "serve");
        return file;
    }

    explicit Server(const ServeOptions& options)
        : options_(options), mapped_(options.model_path), file_(GgufFile::parse(mapped_.bytes())), tokenizer_(file_),
          model_(checked(file_, options), /*repack=*/!options.gpu) {
        const ModelConfig& c = model_.config();
        context_ = context_for(options, c);
        if (options.gpu) {
#if BRISK_METAL
            engine_ = std::make_unique<MetalSession>(model_, mapped_.bytes(), context_);
#else
            throw std::runtime_error("this build has no GPU backend");
#endif
        } else {
            engine_ = std::make_unique<Session>(model_, options.threads, context_);
        }
        name_ = options.name.empty() ? basename_of(options.model_path) : options.name;
        if (options.prefetch) {
            // On a Pi the first request would otherwise spend 10+ seconds paging the
            // weights in from the SD card, which looks like a broken server.
            const auto start = std::chrono::steady_clock::now();
            const std::size_t bytes = mapped_.prefetch(options.lock);
            std::fprintf(stderr, "brisk serve: %.2f GB of weights paged in%s in %.1f s\n",
                         static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0), options.lock ? " and locked" : "",
                         seconds_since(start));
            // One token through the model: the caches and scratch are touched for the
            // first time here rather than in the first request.
            const auto warm = std::chrono::steady_clock::now();
            const Token first = tokenizer_.bos() >= 0 ? tokenizer_.bos() : Token{0};
            engine_->eval(first);
            engine_->rollback(0);
            std::fprintf(stderr, "brisk serve: warm-up pass in %.1f s\n", seconds_since(warm));
        }
    }

    int run() {
        httplib::Server http;
        http.Get("/health", [](const httplib::Request&, httplib::Response& res) {
            res.set_content(R"({"status":"ok"})", "application/json");
        });
        http.Get("/v1/models", [this](const httplib::Request&, httplib::Response& res) {
            const json body = {{"object", "list"},
                               {"data", {{{"id", name_}, {"object", "model"}, {"owned_by", "brisk"}}}}};
            res.set_content(body.dump(), "application/json");
        });
        http.Post("/v1/chat/completions", [this](const httplib::Request& req, httplib::Response& res) {
            handle(req, res, /*chat=*/true);
        });
        http.Post("/v1/completions", [this](const httplib::Request& req, httplib::Response& res) {
            handle(req, res, /*chat=*/false);
        });
        http.set_exception_handler([](const httplib::Request&, httplib::Response& res, std::exception_ptr ep) {
            std::string what = "unknown error";
            try {
                if (ep) std::rethrow_exception(ep);
            } catch (const std::exception& e) {
                what = e.what();
            }
            error(res, 500, what);
        });
        std::fprintf(stderr, "brisk serve: %s on http://%s:%d (context %zu tokens, %s)\n", name_.c_str(),
                     options_.host.c_str(), options_.port, context_, options_.gpu ? "GPU" : "CPU");
        std::signal(SIGINT, on_signal);
        std::signal(SIGTERM, on_signal);
        // The listener is stopped from another thread once a signal arrives; httplib
        // then lets the request in flight finish (its generation stops at the next
        // token, see generate) and returns from listen.
        std::thread stopper([&http] {
            while (!g_stopping.load()) std::this_thread::sleep_for(std::chrono::milliseconds(50));
            http.stop();
        });
        const bool ok = http.listen(options_.host, options_.port);
        g_stopping.store(true);
        stopper.join();
        if (!ok && !g_stopping.load()) {
            throw std::runtime_error("cannot listen on " + options_.host + ":" + std::to_string(options_.port));
        }
        std::fprintf(stderr, "brisk serve: stopped\n");
        return 0;
    }

private:
    static void error(httplib::Response& res, int status, const std::string& message) {
        res.status = status;
        res.set_content(json{{"error", {{"message", message}, {"type", status == 400 ? "invalid_request_error" : "server_error"}}}}.dump(),
                        "application/json");
    }

    Request parse(const json& body, bool chat) const {
        Request r;
        std::string text;
        if (chat) {
            ChatOptions co;
            std::vector<ChatMessage> messages;
            text = render_chat_request(file_, body, messages, co);
            r.tools = !co.tools.empty();
            for (const std::string& t : co.tools) {
                const json tj = json::parse(t);
                const json& f = tj.contains("function") ? tj["function"] : tj;
                if (f.contains("name") && f["name"].is_string()) r.tool_names.push_back(f["name"].get<std::string>());
            }
            r.hold_content = r.tools && chat_format_of(file_) == ChatFormat::Llama3;
        } else {
            if (!body.contains("prompt") || !body["prompt"].is_string()) throw std::invalid_argument("'prompt' must be a string");
            text = body["prompt"].get<std::string>();
        }
        r.prompt = tokenizer_.encode(text);
        if (tokenizer_.bos() >= 0) r.prompt.insert(r.prompt.begin(), tokenizer_.bos());
        if (r.prompt.empty()) throw std::invalid_argument("the prompt is empty");

        if (r.prompt.size() + 1 > context_) {
            throw std::invalid_argument("the prompt has " + std::to_string(r.prompt.size()) + " tokens; the context holds " +
                                        std::to_string(context_));
        }
        const std::size_t room = context_ - r.prompt.size();
        std::size_t max_tokens = room;
        if (body.contains("max_completion_tokens") && body["max_completion_tokens"].is_number()) {
            max_tokens = body["max_completion_tokens"].get<std::size_t>();
        } else if (body.contains("max_tokens") && body["max_tokens"].is_number()) {
            max_tokens = body["max_tokens"].get<std::size_t>();
        }
        r.max_tokens = std::min(max_tokens, room);
        // Sampling defaults when the client sends none: a mild temperature, as the API does.
        r.sampler.temperature = body.value("temperature", 0.7f);
        r.sampler.top_p = body.value("top_p", 0.95f);
        r.sampler.top_k = body.value("top_k", 20u);
        if (body.contains("seed") && body["seed"].is_number()) r.sampler.seed = body["seed"].get<std::uint64_t>();
        else r.sampler.seed = static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
        if (body.contains("stop")) {
            if (body["stop"].is_string()) r.stop.push_back(body["stop"].get<std::string>());
            else if (body["stop"].is_array()) {
                for (const json& s : body["stop"]) {
                    if (s.is_string()) r.stop.push_back(s.get<std::string>());
                }
            }
        }
        r.stream = body.value("stream", false);
        // reasoning_format: "separate" (the default, as llama-server and the DeepSeek API do) or "inline".
        r.separate_reasoning = body.value("reasoning_format", "separate") != "inline";
        return r;
    }

    // Evaluates the prompt, reusing whatever prefix the session already holds
    // (the previous turns of the same conversation, typically), and returns
    // the logits for the next token.
    std::span<const float> prefill(const std::vector<Token>& prompt, std::size_t& reused) {
        std::size_t common = 0;
        while (common < prompt.size() && common < cached_.size() && prompt[common] == cached_[common]) ++common;
        // The logits after the last prompt token are needed, so at least that
        // token is evaluated again.
        if (common == prompt.size()) --common;
        engine_->rollback(common);
        cached_.assign(prompt.begin(), prompt.begin() + static_cast<std::ptrdiff_t>(common));
        reused = common;
        const std::span<const float> logits = engine_->eval(std::span<const Token>(prompt).subspan(common));
        cached_ = prompt;
        return logits;
    }

    struct Generated {
        std::string text;       // the answer
        std::string reasoning;  // the <think> block's text, when separated
        std::vector<std::string> calls;  // the <tool_call> blocks' text (Qwen)
        std::size_t tokens = 0;
        std::string finish_reason = "stop";
    };

    // The tool calls an answer made, as OpenAI's tool_calls; Qwen's come from the <tool_call>
    // blocks, Llama 3's answer is the call itself ({"name": ..., "parameters": {...}}).
    // Whatever does not parse, or names a function that was not offered, stays in the content.
    static json tool_calls_of(Generated& g, const Request& r, const std::string& id) {
        json calls = json::array();
        auto add = [&](const std::string& text, const char* args_key) {
            json j = json::parse(text, nullptr, false);
            if (!j.is_object() || !j.contains("name") || !j["name"].is_string()) return false;
            if (std::find(r.tool_names.begin(), r.tool_names.end(), j["name"].get<std::string>()) == r.tool_names.end()) return false;
            const json args = j.contains(args_key) ? j[args_key] : json::object();
            if (!args.is_object() && !args.is_string()) return false;
            calls.push_back({{"id", id + "-" + std::to_string(calls.size())},
                             {"type", "function"},
                             {"function", {{"name", j["name"]}, {"arguments", args.is_string() ? args.get<std::string>() : args.dump()}}}});
            return true;
        };
        for (const std::string& c : g.calls) {
            if (c.starts_with("call:")) {  // Gemma 4: call:name{key:<|"|>value<|"|>,...}
                const std::size_t brace = c.find('{');
                const std::string args = brace == std::string::npos ? "" : gemma4_arguments_to_json(std::string_view(c).substr(brace));
                if (!args.empty() && add("{\"name\": " + json(c.substr(5, brace - 5)).dump() + ", \"arguments\": " + args + "}", "arguments")) continue;
                g.text += std::string(g.text.empty() ? "" : "\n") + "<|tool_call>" + c + "<tool_call|>";
                continue;
            }
            if (!add(c, "arguments")) g.text += std::string(g.text.empty() ? "" : "\n") + "<tool_call>\n" + c + "\n</tool_call>";
        }
        if (r.hold_content && calls.empty()) {
            const std::string_view t = trim(g.text);
            if (t.starts_with("{") && add(std::string(t), "parameters")) g.text.clear();
        }
        return calls;
    }

    // Generates for one request, calling emit(text, reasoning) with each piece of
    // decoded text (reasoning: the piece belongs to a <think> block); emit returns
    // false to stop (client gone). Tool-call blocks are collected, not emitted.
    // Holds the session lock.
    template <typename Emit>
    Generated generate(const Request& r, Emit emit, double& prompt_seconds, double& gen_seconds, std::size_t& reused) {
        Generated g;
        const auto prompt_start = std::chrono::steady_clock::now();
        std::span<const float> logits = prefill(r.prompt, reused);
        prompt_seconds = seconds_since(prompt_start);
        const auto gen_start = std::chrono::steady_clock::now();
        Sampler sampler(r.sampler);
        std::string pending;  // decoded text not yet emitted (incomplete UTF-8, or a possible stop-string prefix)
        std::size_t longest_stop = 0;
        for (const std::string& s : r.stop) longest_stop = std::max(longest_stop, s.size());
        bool stopped_by_string = false;
        bool reasoning = false;     // inside a <think> block (Qwen3, DeepSeek-R1: the tags are single tokens)
        bool call = false;          // inside a <tool_call> block (Qwen: also single tokens)
        bool skip_newlines = false; // the newlines the templates put after the tags are dropped
        bool skip_channel_name = false;  // Gemma 4: "thought\n" after <|channel>
        auto flush = [&](const std::string& piece) {
            if (call) {
                g.calls.back() += piece;
                return true;
            }
            (reasoning ? g.reasoning : g.text) += piece;
            return r.hold_content && !reasoning ? true : emit(piece, reasoning);
        };
        while (g.tokens < r.max_tokens) {
            if (g_stopping.load()) {  // shutting down: end this answer here, as a truncation
                g.finish_reason = "length";
                break;
            }
            const Token next = sampler.sample(logits);
            if (tokenizer_.is_end_of_generation(next)) break;
            // Gemma 4 asks for the result after a call: its turn is over.
            if (!g.calls.empty() && !call && tokenizer_.piece(next) == "<|tool_response>") break;
            ++g.tokens;
            cached_.push_back(next);
            const std::string_view piece_text = tokenizer_.piece(next);
            // Qwen3/DeepSeek: <think>...</think>. Gemma 4: <|channel>thought\n...<channel|> (the name after the
            // opening token is dropped up to its newline).
            const bool think_tag = r.separate_reasoning && (piece_text == "<think>" || piece_text == "</think>" ||
                                                            piece_text == "<|channel>" || piece_text == "<channel|>");
            const bool call_tag = r.tools && (piece_text == "<tool_call>" || piece_text == "</tool_call>" ||
                                              piece_text == "<|tool_call>" || piece_text == "<tool_call|>");
            if (think_tag || call_tag) {
                if (!pending.empty()) {  // whatever was held back belongs to the block that just ended
                    if (!flush(pending)) {
                        g.finish_reason = "abort";
                        break;
                    }
                    pending.clear();
                }
                if (think_tag) {
                    reasoning = piece_text == "<think>" || piece_text == "<|channel>";
                    skip_channel_name = piece_text == "<|channel>";
                }
                if (call_tag) {
                    call = piece_text == "<tool_call>" || piece_text == "<|tool_call>";
                    if (call) g.calls.emplace_back();
                }
                skip_newlines = true;
                if (g.tokens < r.max_tokens) logits = engine_->eval(next);
                continue;
            }
            if (!tokenizer_.is_control(next) || call) {  // a call keeps its control tokens (Gemma 4's <|"|> string marks)
                std::string_view text = piece_text;
                if (skip_channel_name) {
                    const std::size_t nl = text.find('\n');
                    if (nl == std::string_view::npos) {
                        text = {};
                    } else {
                        text.remove_prefix(nl + 1);
                        skip_channel_name = false;
                    }
                }
                if (skip_newlines) {
                    while (!text.empty() && text.front() == '\n') text.remove_prefix(1);
                    skip_newlines = text.empty();
                }
                pending.append(text);
            }
            // Stop strings: the last longest_stop - 1 bytes are always held back,
            // so a stop string can only appear within `pending`.
            for (const std::string& s : r.stop) {
                const std::size_t at = pending.find(s);
                if (at != std::string::npos) {
                    pending.resize(at);
                    stopped_by_string = true;
                }
            }
            std::size_t ready = complete_utf8_prefix(pending);
            if (!stopped_by_string && longest_stop > 1) {
                ready = std::min(ready, pending.size() >= longest_stop - 1 ? pending.size() - (longest_stop - 1) : 0);
            }
            if (ready > 0) {
                const std::string piece = pending.substr(0, ready);
                pending.erase(0, ready);
                if (!flush(piece)) {
                    g.finish_reason = "abort";
                    break;
                }
            }
            if (stopped_by_string) break;
            if (g.tokens < r.max_tokens) logits = engine_->eval(next);
        }
        if (!pending.empty() && g.finish_reason != "abort") flush(pending);
        for (std::string& c : g.calls) c = std::string(trim(c));
        if (g.finish_reason != "abort" && !g_stopping.load()) {
            g.finish_reason = stopped_by_string || g.tokens < r.max_tokens ? "stop" : "length";
        }
        gen_seconds = seconds_since(gen_start);
        return g;
    }

    static std::string new_id(bool chat) {
        return std::string(chat ? "chatcmpl-" : "cmpl-") +
               std::to_string(std::chrono::system_clock::now().time_since_epoch().count() % 1000000000);
    }

    json chunk(const std::string& id, bool chat, const json& delta_or_text, const std::optional<std::string>& finish) const {
        json choice = {{"index", 0}, {"finish_reason", finish ? json(*finish) : json()}};
        if (chat) choice["delta"] = delta_or_text;
        else choice["text"] = delta_or_text;
        return json{{"id", id},
                    {"object", chat ? "chat.completion.chunk" : "text_completion"},
                    {"created", std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count()},
                    {"model", name_},
                    {"choices", {choice}}};
    }

    void log(const Request& r, const Generated& g, double prompt_seconds, double gen_seconds, std::size_t reused) const {
        const std::size_t fresh = r.prompt.size() - reused;
        std::fprintf(stderr, "request: prompt %zu tokens (%zu reused, %zu evaluated in %.2f s) | generated %zu tokens, %.1f tokens/s | %s\n",
                     r.prompt.size(), reused, fresh, prompt_seconds, g.tokens,
                     gen_seconds > 0 ? static_cast<double>(g.tokens) / gen_seconds : 0.0, g.finish_reason.c_str());
    }

    void handle(const httplib::Request& req, httplib::Response& res, bool chat) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (const json::exception& e) {
            return error(res, 400, std::string("invalid JSON: ") + e.what());
        }
        Request r;
        try {
            r = parse(body, chat);
        } catch (const std::exception& e) {
            return error(res, 400, e.what());
        }
        const std::string id = new_id(chat);

        if (!r.stream) {
            std::lock_guard<std::mutex> lock(mutex_);
            double prompt_seconds = 0, gen_seconds = 0;
            std::size_t reused = 0;
            Generated g = generate(r, [](const std::string&, bool) { return true; }, prompt_seconds, gen_seconds, reused);
            if (options_.verbose) log(r, g, prompt_seconds, gen_seconds, reused);
            const json calls = chat ? tool_calls_of(g, r, id) : json::array();
            if (!calls.empty() && g.finish_reason == "stop") g.finish_reason = "tool_calls";
            json choice = {{"index", 0}, {"finish_reason", g.finish_reason}};
            if (chat) {
                choice["message"] = {{"role", "assistant"}, {"content", g.text}};
                if (!g.reasoning.empty()) choice["message"]["reasoning_content"] = g.reasoning;
                if (!calls.empty()) choice["message"]["tool_calls"] = calls;
            } else {
                choice["text"] = g.reasoning.empty() ? g.text : g.reasoning + g.text;
            }
            const json out = {{"id", id},
                              {"object", chat ? "chat.completion" : "text_completion"},
                              {"created", std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count()},
                              {"model", name_},
                              {"choices", {choice}},
                              {"usage", {{"prompt_tokens", r.prompt.size()}, {"completion_tokens", g.tokens},
                                         {"total_tokens", r.prompt.size() + g.tokens}}}};
            res.set_content(out.dump(), "application/json");
            return;
        }

        // Streaming: the whole generation runs inside the content provider, one
        // SSE event per decoded piece, then the finish chunk and [DONE].
        res.set_header("Cache-Control", "no-cache");
        res.set_chunked_content_provider("text/event-stream", [this, r, id, chat](std::size_t, httplib::DataSink& sink) {
            std::lock_guard<std::mutex> lock(mutex_);
            auto send = [&](const json& j) {
                const std::string line = "data: " + j.dump() + "\n\n";
                return sink.write(line.data(), line.size());
            };
            if (chat) send(chunk(id, chat, json{{"role", "assistant"}, {"content", ""}}, std::nullopt));
            double prompt_seconds = 0, gen_seconds = 0;
            std::size_t reused = 0;
            Generated g = generate(
                r,
                [&](const std::string& piece, bool reasoning) {
                    const json delta = chat ? json{{reasoning ? "reasoning_content" : "content", piece}} : json(piece);
                    return send(chunk(id, chat, delta, std::nullopt));
                },
                prompt_seconds, gen_seconds, reused);
            if (options_.verbose) log(r, g, prompt_seconds, gen_seconds, reused);
            if (chat && g.finish_reason != "abort") {
                // The calls (and, for Llama 3, the held-back content) go out once the answer is known.
                const std::size_t streamed = r.hold_content ? 0 : g.text.size();
                json calls = tool_calls_of(g, r, id);
                if (g.text.size() > streamed) send(chunk(id, chat, json{{"content", g.text.substr(streamed)}}, std::nullopt));
                if (!calls.empty()) {
                    for (std::size_t i = 0; i < calls.size(); ++i) calls[i]["index"] = i;
                    send(chunk(id, chat, json{{"tool_calls", calls}}, std::nullopt));
                    if (g.finish_reason == "stop") g.finish_reason = "tool_calls";
                }
            }
            if (g.finish_reason != "abort") {
                json last = chunk(id, chat, chat ? json::object() : json(""), g.finish_reason);
                last["usage"] = {{"prompt_tokens", r.prompt.size()}, {"completion_tokens", g.tokens},
                                 {"total_tokens", r.prompt.size() + g.tokens}};
                send(last);
                const std::string done = "data: [DONE]\n\n";
                sink.write(done.data(), done.size());
            }
            sink.done();
            return true;
        });
    }

    ServeOptions options_;
    MappedFile mapped_;
    GgufFile file_;
    Tokenizer tokenizer_;
    Model model_;
    std::unique_ptr<Engine> engine_;
    std::size_t context_ = 0;
    std::string name_;
    std::mutex mutex_;           // one generation at a time: the session is shared
    std::vector<Token> cached_;  // the tokens the session currently holds
};

}  // namespace

int serve(const ServeOptions& options) {
    Server server(options);
    return server.run();
}

}  // namespace brisk
