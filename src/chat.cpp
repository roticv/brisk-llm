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

void check_role(const ChatMessage& m, bool tools = false) {
    if (m.role == "tool" && tools) return;
    if (m.role != "system" && m.role != "user" && m.role != "assistant") {
        throw std::runtime_error("chat: unknown role '" + m.role + "'");
    }
}

// Qwen's rendering of a call: {"name": "...", "arguments": {...}}
std::string qwen_tool_call(const ToolCall& c) { return "{\"name\": \"" + c.name + "\", \"arguments\": " + c.arguments + "}"; }

// ChatML (Qwen2.5, Qwen3): every message is <|im_start|>role\ncontent<|im_end|>\n.
std::string render_chatml(const std::vector<ChatMessage>& messages, const ChatOptions& o, bool qwen3) {
    std::string out;
    std::size_t first = 0;
    if (!o.tools.empty()) {
        // Qwen's tools block: the system message (if any) then the function signatures and the call format.
        out += "<|im_start|>system\n";
        if (!messages.empty() && messages[0].role == "system") {
            out += messages[0].content + "\n\n";
            first = 1;
        }
        out += "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
               "You are provided with function signatures within <tools></tools> XML tags:\n<tools>";
        for (const std::string& t : o.tools) out += "\n" + t;
        out += "\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
               "<tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
               "</tool_call><|im_end|>\n";
    }
    for (std::size_t i = first; i < messages.size(); ++i) {
        const ChatMessage& m = messages[i];
        check_role(m, true);
        if (m.role == "tool") {
            // Consecutive results share one user turn.
            if (i == first || messages[i - 1].role != "tool") out += "<|im_start|>user";
            out += "\n<tool_response>\n" + m.content + "\n</tool_response>";
            if (i + 1 == messages.size() || messages[i + 1].role != "tool") out += "<|im_end|>\n";
            continue;
        }
        out += "<|im_start|>" + m.role + "\n";
        const std::string content = qwen3 && m.role == "assistant" ? std::string(after_think(m.content)) : m.content;
        out += content;
        for (std::size_t c = 0; c < m.tool_calls.size(); ++c) {
            if (c > 0 || !content.empty()) out += "\n";
            out += "<tool_call>\n" + qwen_tool_call(m.tool_calls[c]) + "\n</tool_call>";
        }
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
    std::size_t first = 0;
    if (!o.tools.empty()) {
        // Llama 3's tools in the system message (the template's tools_in_user_message = false form).
        out += "<|start_header_id|>system<|end_header_id|>\n\nEnvironment: ipython\nCutting Knowledge Date: December 2023\n\n"
               "You have access to the following functions. To call a function, please respond with JSON for a function call."
               "Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}."
               "Do not use variables.\n\n";
        for (const std::string& t : o.tools) out += t + "\n\n";
        if (!messages.empty() && messages[0].role == "system") {
            out += std::string(trim(messages[0].content));
            first = 1;
        }
        out += "<|eot_id|>";
    }
    for (std::size_t i = first; i < messages.size(); ++i) {
        const ChatMessage& m = messages[i];
        check_role(m, true);
        if (!m.tool_calls.empty()) {
            if (m.tool_calls.size() != 1) throw std::runtime_error("chat: Llama 3 makes one tool call at a time");
            out += "<|start_header_id|>assistant<|end_header_id|>\n\n{\"name\": \"" + m.tool_calls[0].name + "\", \"parameters\": " +
                   m.tool_calls[0].arguments + "}<|eot_id|>";
            continue;
        }
        const std::string role = m.role == "tool" ? "ipython" : m.role;
        out += "<|start_header_id|>" + role + "<|end_header_id|>\n\n" + std::string(trim(m.content)) + "<|eot_id|>";
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

// Mistral's [INST] format (v0.3), as llama.cpp renders it: the system message
// precedes the first user message with a newline; "[INST] text [/INST]" per
// user turn, the answer followed by </s>. The BOS token comes from the tokenizer.
std::string render_mistral(const std::vector<ChatMessage>& messages, const ChatOptions&) {
    std::string out, pending_system;
    for (std::size_t i = 0; i < messages.size(); ++i) {
        const ChatMessage& m = messages[i];
        check_role(m);
        if (m.role == "system") {
            if (i != 0) throw std::runtime_error("chat: Mistral takes a system message only first");
            pending_system = m.content;
            continue;
        }
        if (m.role == "user") {
            out += "[INST] ";
            if (!pending_system.empty()) {
                out += pending_system + "\n";
                pending_system.clear();
            }
            out += m.content + " [/INST]";
        } else {
            out += m.content + "</s>";
        }
    }
    return out;
}

// DeepSeek R1 (distills): the system message as plain text, then
// <｜User｜>text<｜Assistant｜>answer<｜end▁of▁sentence｜> per exchange.
std::string render_deepseek(const std::vector<ChatMessage>& messages, const ChatOptions&) {
    std::string out;
    for (std::size_t i = 0; i < messages.size(); ++i) {
        const ChatMessage& m = messages[i];
        check_role(m);
        if (m.role == "system") {
            if (i != 0) throw std::runtime_error("chat: DeepSeek takes a system message only first");
            out += m.content + "\n\n";
        } else if (m.role == "user") {
            out += "<｜User｜>" + m.content + "<｜Assistant｜>";
        } else {
            out += m.content + "<｜end▁of▁sentence｜>";
        }
    }
    return out;
}

// Phi-4-mini: <|role|>content<|end|> per message, as its file's template has it
// (llama.cpp's built-in renderer adds Phi-3's newlines; the file does not).
std::string render_phi(const std::vector<ChatMessage>& messages, const ChatOptions& o) {
    std::string out;
    for (const ChatMessage& m : messages) {
        check_role(m);
        out += "<|" + m.role + "|>" + m.content + "<|end|>";
    }
    if (o.generation_prompt) out += "<|assistant|>";
    return out;
}

}  // namespace

ChatFormat chat_format_of(const GgufFile& file) {
    const std::string_view tmpl = file.get<std::string_view>("tokenizer.chat_template").value_or("");
    if (tmpl.find("<|im_start|>") != std::string_view::npos) {
        return tmpl.find("<think>") != std::string_view::npos ? ChatFormat::Qwen3 : ChatFormat::ChatML;
    }
    if (tmpl.find("<|start_header_id|>") != std::string_view::npos) return ChatFormat::Llama3;
    if (tmpl.find("<start_of_turn>") != std::string_view::npos) return ChatFormat::Gemma3;
    if (tmpl.find("<|turn>") != std::string_view::npos) return ChatFormat::Gemma4;
    if (tmpl.find("[INST]") != std::string_view::npos) return ChatFormat::Mistral;
    if (tmpl.find("<|assistant|>") != std::string_view::npos && tmpl.find("<|end|>") != std::string_view::npos) return ChatFormat::Phi;
    if (tmpl.find("<｜User｜>") != std::string_view::npos) return ChatFormat::DeepSeek;
    if (tmpl.find("capitalize") != std::string_view::npos) return ChatFormat::BitNet;
    switch (architecture_of(file)) {  // no template, or an unknown one: the family's usual format
        case Architecture::Qwen3: return ChatFormat::Qwen3;
        case Architecture::Qwen2: return ChatFormat::ChatML;
        case Architecture::Llama: return ChatFormat::Llama3;
        case Architecture::Gemma3: return ChatFormat::Gemma3;
        case Architecture::Gemma4: return ChatFormat::Gemma4;
        case Architecture::BitNet: return ChatFormat::BitNet;
        case Architecture::Phi3: return ChatFormat::Phi;
        case Architecture::LFM2: return ChatFormat::ChatML;
    }
    throw std::runtime_error("chat: no template for this file");
}

bool chat_format_has_tools(ChatFormat format) {
    return format == ChatFormat::ChatML || format == ChatFormat::Qwen3 || format == ChatFormat::Llama3;
}

std::string render_chat(ChatFormat format, const std::vector<ChatMessage>& messages, const ChatOptions& options) {
    if (!options.tools.empty() && !chat_format_has_tools(format)) {
        throw std::runtime_error("chat: tools are not supported for this model's template (Qwen and Llama 3 files take them)");
    }
    for (const ChatMessage& m : messages) {
        if ((m.role == "tool" || !m.tool_calls.empty()) && !chat_format_has_tools(format)) {
            throw std::runtime_error("chat: tool messages are not supported for this model's template");
        }
    }
    switch (format) {
        case ChatFormat::Qwen3: return render_chatml(messages, options, true);
        case ChatFormat::ChatML: return render_chatml(messages, options, false);
        case ChatFormat::Llama3: return render_llama3(messages, options);
        case ChatFormat::Gemma3: return render_gemma3(messages, options);
        case ChatFormat::Gemma4: return render_gemma4(messages, options);
        case ChatFormat::Mistral: return render_mistral(messages, options);
        case ChatFormat::DeepSeek: return render_deepseek(messages, options);
        case ChatFormat::Phi: return render_phi(messages, options);
        case ChatFormat::BitNet: return render_bitnet(messages, options);
    }
    throw std::runtime_error("chat: unknown format");
}

}  // namespace brisk
