#include "chat.h"

#include <stdexcept>
#include <string_view>

namespace brisk {

namespace {

std::string_view trim(std::string_view s) {
    const auto is_space = [](char c) { return c == ' ' || c == '\n' || c == '\t' || c == '\r'; };
    while (!s.empty() && is_space(s.front())) s.remove_prefix(1);
    while (!s.empty() && is_space(s.back())) s.remove_suffix(1);
    return s;
}

// Qwen's content with a reasoning block removed: the templates keep only the
// text after </think> of earlier assistant turns.
std::string_view after_think(std::string_view content) {
    const std::size_t end = content.rfind("</think>");
    if (end == std::string_view::npos) return content;
    std::string_view rest = content.substr(end + 8);
    while (!rest.empty() && rest.front() == '\n') rest.remove_prefix(1);
    return rest;
}

void check_role(const ChatMessage& m) {
    if (m.role != "system" && m.role != "user" && m.role != "assistant") {
        throw std::runtime_error("chat: unknown role '" + m.role + "'");
    }
}

// ChatML (Qwen2.5, Qwen3): every message is <|im_start|>role\ncontent<|im_end|>\n.
std::string render_chatml(const std::vector<ChatMessage>& messages, const ChatOptions& o, bool qwen3) {
    std::string out;
    for (const ChatMessage& m : messages) {
        check_role(m);
        out += "<|im_start|>" + m.role + "\n";
        out += qwen3 && m.role == "assistant" ? std::string(after_think(m.content)) : m.content;
        out += "<|im_end|>\n";
    }
    if (o.generation_prompt) {
        out += "<|im_start|>assistant\n";
        if (qwen3 && !o.think) out += "<think>\n\n</think>\n\n";
    }
    return out;
}

// Llama 3: <|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>.
std::string render_llama3(const std::vector<ChatMessage>& messages, const ChatOptions& o) {
    std::string out;
    for (const ChatMessage& m : messages) {
        check_role(m);
        out += "<|start_header_id|>" + m.role + "<|end_header_id|>\n\n" + std::string(trim(m.content)) + "<|eot_id|>";
    }
    if (o.generation_prompt) out += "<|start_header_id|>assistant<|end_header_id|>\n\n";
    return out;
}

// Gemma 3: <start_of_turn>user|model\ncontent<end_of_turn>\n; a leading system
// message is prepended to the first user turn with a blank line.
std::string render_gemma3(const std::vector<ChatMessage>& messages, const ChatOptions& o) {
    std::string out, pending_system;
    for (std::size_t i = 0; i < messages.size(); ++i) {
        const ChatMessage& m = messages[i];
        check_role(m);
        if (m.role == "system") {
            if (i != 0) throw std::runtime_error("chat: Gemma 3 takes a system message only first");
            pending_system = std::string(trim(m.content));
            continue;
        }
        const std::string role = m.role == "assistant" ? "model" : "user";
        out += "<start_of_turn>" + role + "\n";
        if (!pending_system.empty() && m.role == "user") {
            out += pending_system + "\n\n";
            pending_system.clear();
        }
        out += std::string(trim(m.content)) + "<end_of_turn>\n";
    }
    if (o.generation_prompt) out += "<start_of_turn>model\n";
    return out;
}

// Gemma 4: <|turn>system|user|model\ncontent<turn|>\n; the system message is
// its own turn. (No thinking: the template's enable_thinking is off by default.)
std::string render_gemma4(const std::vector<ChatMessage>& messages, const ChatOptions& o) {
    std::string out;
    for (std::size_t i = 0; i < messages.size(); ++i) {
        const ChatMessage& m = messages[i];
        check_role(m);
        if (m.role == "system" && i != 0) throw std::runtime_error("chat: Gemma 4 takes a system message only first");
        const std::string role = m.role == "assistant" ? "model" : m.role;
        out += "<|turn>" + role + "\n" + std::string(trim(m.content)) + "<turn|>\n";
    }
    if (o.generation_prompt) out += "<|turn>model\n";
    return out;
}

// BitNet b1.58 2B4T: "Role: content<|eot_id|>" per message, "Assistant: " to prompt.
std::string render_bitnet(const std::vector<ChatMessage>& messages, const ChatOptions& o) {
    std::string out;
    for (const ChatMessage& m : messages) {
        check_role(m);
        std::string role = m.role;
        role[0] = static_cast<char>(role[0] - 'a' + 'A');
        out += role + ": " + std::string(trim(m.content)) + "<|eot_id|>";
    }
    if (o.generation_prompt) out += "Assistant: ";
    return out;
}

}  // namespace

std::string render_chat(Architecture architecture, const std::vector<ChatMessage>& messages, const ChatOptions& options) {
    switch (architecture) {
        case Architecture::Qwen3: return render_chatml(messages, options, true);
        case Architecture::Qwen2: return render_chatml(messages, options, false);
        case Architecture::Llama: return render_llama3(messages, options);
        case Architecture::Gemma3: return render_gemma3(messages, options);
        case Architecture::Gemma4: return render_gemma4(messages, options);
        case Architecture::BitNet: return render_bitnet(messages, options);
    }
    throw std::runtime_error("chat: no template for this architecture");
}

}  // namespace brisk
