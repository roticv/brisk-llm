#include "chat.h"

#include "json.hpp"

#include <algorithm>
#include <cctype>
#include <ctime>
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
std::string_view strip_newlines(std::string_view s, bool front = true, bool back = true) {
    while (front && !s.empty() && s.front() == '\n') s.remove_prefix(1);
    while (back && !s.empty() && s.back() == '\n') s.remove_suffix(1);
    return s;
}

std::string_view after_think(std::string_view content) {
    const std::size_t end = content.rfind("</think>");
    if (end == std::string_view::npos) return content;
    return strip_newlines(content.substr(end + 8), true, false);
}

// The reasoning of an earlier Qwen3 answer: its reasoning_content, or the text
// of its <think> block (the template's split on </think>, then on <think>).
std::string qwen3_reasoning(const ChatMessage& m) {
    if (!m.reasoning.empty()) return m.reasoning;
    const std::size_t end = m.content.rfind("</think>");
    if (end == std::string_view::npos) return "";
    std::string_view r = strip_newlines(std::string_view(m.content).substr(0, end), false, true);
    const std::size_t open = r.rfind("<think>");
    if (open != std::string_view::npos) r = strip_newlines(r.substr(open + 7), true, false);
    return std::string(r);
}

void check_role(const ChatMessage& m, bool tools = false) {
    if (m.role == "tool" && tools) return;
    if (m.role != "system" && m.role != "user" && m.role != "assistant") {
        throw std::runtime_error("chat: unknown role '" + m.role + "'");
    }
}

using ojson = nlohmann::ordered_json;

// JSON as Jinja's tojson filter writes it (Python's json.dumps): insertion order,
// ", " and ": " separators, and with indent, one entry per line.
void python_json(const ojson& j, int indent, int level, std::string& out) {
    const std::string nl = indent ? "\n" + std::string(static_cast<std::size_t>(indent * (level + 1)), ' ') : "";
    const std::string end = indent ? "\n" + std::string(static_cast<std::size_t>(indent * level), ' ') : "";
    if (j.is_object()) {
        if (j.empty()) {
            out += "{}";
            return;
        }
        out += "{";
        bool first = true;
        for (const auto& [key, value] : j.items()) {
            out += (first ? "" : (indent ? "," : ", ")) + nl + ojson(key).dump() + ": ";
            python_json(value, indent, level + 1, out);
            first = false;
        }
        out += end + "}";
    } else if (j.is_array()) {
        if (j.empty()) {
            out += "[]";
            return;
        }
        out += "[";
        bool first = true;
        for (const ojson& value : j) {
            out += (first ? "" : (indent ? "," : ", ")) + nl;
            python_json(value, indent, level + 1, out);
            first = false;
        }
        out += end + "]";
    } else {
        out += j.dump();
    }
}

std::string python_json(const std::string& text, int indent = 0) {
    const ojson j = ojson::parse(text, nullptr, false);
    if (j.is_discarded()) return text;  // not JSON: as given
    std::string out;
    python_json(j, indent, 0, out);
    return out;
}

// Qwen's rendering of a call: {"name": "...", "arguments": {...}}
std::string qwen_tool_call(const ToolCall& c) {
    return "{\"name\": \"" + c.name + "\", \"arguments\": " + python_json(c.arguments) + "}";
}

// Llama 3's "Today Date: 26 Oct 2026" line.
std::string today() {
    const std::time_t now = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof buf, "%d %b %Y", std::localtime(&now));
    return buf;
}

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
        } else if (!o.default_system.empty()) {
            out += o.default_system + "\n\n";
        }
        out += "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
               "You are provided with function signatures within <tools></tools> XML tags:\n<tools>";
        for (const std::string& t : o.tools) out += "\n" + python_json(t);
        out += "\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
               "<tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
               "</tool_call><|im_end|>\n";
    } else if (!o.default_system.empty() && (messages.empty() || messages[0].role != "system")) {
        out += "<|im_start|>system\n" + o.default_system + "<|im_end|>\n";
    }
    // Qwen3 keeps the reasoning of the answers after the last real user query (the
    // turns of a tool loop); a user turn that is only a <tool_response> block does not count.
    std::size_t last_query = messages.empty() ? 0 : messages.size() - 1;
    for (std::size_t i = messages.size(); i-- > 0;) {
        const std::string_view c = messages[i].content;
        if (messages[i].role == "user" && !(c.starts_with("<tool_response>") && c.ends_with("</tool_response>"))) {
            last_query = i;
            break;
        }
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
        std::string content = qwen3 && m.role == "assistant" ? std::string(after_think(m.content)) : m.content;
        if (qwen3 && m.role == "assistant" && i > last_query) {
            const std::string reasoning = qwen3_reasoning(m);
            if (i + 1 == messages.size() || !trim(reasoning).empty()) {
                out += "<think>\n" + std::string(strip_newlines(reasoning)) + "\n</think>\n\n";
                content = std::string(strip_newlines(content, true, false));
            }
        }
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
    // The system header is always there, with the template's dates; the tools go in
    // the first user message (the template's tools_in_user_message default).
    std::string out = "<|start_header_id|>system<|end_header_id|>\n\n";
    if (!o.tools.empty()) out += "Environment: ipython\n";
    out += "Cutting Knowledge Date: December 2023\nToday Date: " + today() + "\n\n";
    std::size_t first = 0;
    if (!messages.empty() && messages[0].role == "system") {
        out += std::string(trim(messages[0].content));
        first = 1;
    }
    out += "<|eot_id|>";
    if (!o.tools.empty()) {
        if (first >= messages.size()) throw std::runtime_error("chat: Llama 3 puts the tools in the first user message, and there is none");
        out += "<|start_header_id|>user<|end_header_id|>\n\n"
               "Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.\n\n"
               "Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.Do not use variables.\n\n";
        for (const std::string& t : o.tools) out += python_json(t, 4) + "\n\n";
        out += std::string(trim(messages[first].content)) + "<|eot_id|>";
        ++first;
    }
    for (std::size_t i = first; i < messages.size(); ++i) {
        const ChatMessage& m = messages[i];
        check_role(m, true);
        if (!m.tool_calls.empty()) {
            if (m.tool_calls.size() != 1) throw std::runtime_error("chat: Llama 3 makes one tool call at a time");
            out += "<|start_header_id|>assistant<|end_header_id|>\n\n{\"name\": \"" + m.tool_calls[0].name + "\", \"parameters\": " +
                   python_json(m.tool_calls[0].arguments) + "}<|eot_id|>";
            continue;
        }
        if (m.role == "tool") {
            // The template passes a result through tojson, which quotes a string (a string is iterable in Jinja).
            out += "<|start_header_id|>ipython<|end_header_id|>\n\n" + ojson(m.content).dump() + "<|eot_id|>";
            continue;
        }
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
// Gemma 4's value syntax (the template's format_argument): JSON with keys in
// sorted order, strings between <|"|> marks, and object keys bare unless escaped.
void gemma4_value(const nlohmann::json& v, bool escape_keys, std::string& out) {  // nlohmann::json keeps keys sorted, as dictsort does
    if (v.is_null()) {
        out += "null";
    } else if (v.is_string()) {
        out += "<|\"|>" + v.get<std::string>() + "<|\"|>";
    } else if (v.is_boolean()) {
        out += v.get<bool>() ? "true" : "false";
    } else if (v.is_object()) {
        out += "{";
        bool first = true;
        for (const auto& [key, value] : v.items()) {
            out += first ? "" : ",";
            first = false;
            out += escape_keys ? "<|\"|>" + key + "<|\"|>" : key;
            out += ":";
            gemma4_value(value, escape_keys, out);
        }
        out += "}";
    } else if (v.is_array()) {
        out += "[";
        for (std::size_t i = 0; i < v.size(); ++i) {
            if (i) out += ",";
            gemma4_value(v[i], escape_keys, out);
        }
        out += "]";
    } else {
        out += v.dump();
    }
}

std::string upper(std::string s) {
    for (char& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
}

std::string gemma4_type(const nlohmann::json& schema) {
    return schema.contains("type") && schema["type"].is_string() ? upper(schema["type"].get<std::string>()) : "";
}

void gemma4_required(const nlohmann::json& list, std::string& out) {
    out += "required:[";
    for (std::size_t i = 0; i < list.size(); ++i) {
        if (i) out += ",";
        out += "<|\"|>" + (list[i].is_string() ? list[i].get<std::string>() : list[i].dump()) + "<|\"|>";
    }
    out += "]";
}

// The template's format_parameters: each property as key:{description,enum/items,nullable,properties,required,type}.
void gemma4_parameters(const nlohmann::json& properties, bool filter_keys, std::string& out) {
    static const char* const kStandard[] = {"description", "type", "properties", "required", "nullable"};
    bool first = true;
    for (const auto& [key, value] : properties.items()) {
        if (filter_keys && std::find(std::begin(kStandard), std::end(kStandard), key) != std::end(kStandard)) continue;
        out += first ? "" : ",";
        first = false;
        out += key + ":{";
        bool comma = false;
        auto part = [&] {
            if (comma) out += ",";
            comma = true;
        };
        const std::string type = value.is_object() ? gemma4_type(value) : "";
        if (value.is_object() && value.contains("description") && !value["description"].is_null() &&
            !(value["description"].is_string() && value["description"].get<std::string>().empty())) {
            part();
            out += "description:<|\"|>" + (value["description"].is_string() ? value["description"].get<std::string>() : value["description"].dump()) + "<|\"|>";
        }
        if (type == "STRING") {
            if (value.contains("enum") && value["enum"].is_array() && !value["enum"].empty()) {
                part();
                out += "enum:";
                gemma4_value(value["enum"], true, out);
            }
        } else if (type == "ARRAY") {
            if (value.contains("items") && value["items"].is_object() && !value["items"].empty()) {
                part();
                out += "items:{";
                bool item_first = true;
                for (const auto& [ik, iv] : value["items"].items()) {
                    if (iv.is_null()) continue;
                    out += item_first ? "" : ",";
                    item_first = false;
                    if (ik == "properties") {
                        out += "properties:{";
                        if (iv.is_object()) gemma4_parameters(iv, false, out);
                        out += "}";
                    } else if (ik == "required") {
                        gemma4_required(iv, out);
                    } else if (ik == "type") {
                        out += "type:";
                        if (iv.is_string()) {
                            gemma4_value(nlohmann::json(upper(iv.get<std::string>())), true, out);
                        } else {
                            nlohmann::json ups = nlohmann::json::array();
                            for (const auto& t : iv) ups.push_back(t.is_string() ? nlohmann::json(upper(t.get<std::string>())) : t);
                            gemma4_value(ups, true, out);
                        }
                    } else {
                        out += ik + ":";
                        gemma4_value(iv, true, out);
                    }
                }
                out += "}";
            }
        }
        if (value.is_object() && value.contains("nullable") && value["nullable"].is_boolean() && value["nullable"].get<bool>()) {
            part();
            out += "nullable:true";
        }
        if (type == "OBJECT") {
            if (value.contains("properties") && value["properties"].is_object()) {
                part();
                out += "properties:{";
                gemma4_parameters(value["properties"], false, out);
                out += "}";
            } else if (value.is_object()) {
                part();
                out += "properties:{";
                gemma4_parameters(value, true, out);
                out += "}";
            }
            if (value.contains("required") && value["required"].is_array() && !value["required"].empty()) {
                part();
                gemma4_required(value["required"], out);
            }
        }
        part();
        out += "type:<|\"|>" + type + "<|\"|>}";
    }
}

// The template's format_function_declaration, for one OpenAI tool definition.
std::string gemma4_declaration(const std::string& tool_json) {
    const nlohmann::json tool = nlohmann::json::parse(tool_json);  // sorted keys, as the template's dictsort
    const nlohmann::json& f = tool.contains("function") ? tool["function"] : tool;
    std::string out = "declaration:" + f.value("name", "") + "{description:<|\"|>" + f.value("description", "") + "<|\"|>";
    if (f.contains("parameters") && f["parameters"].is_object() && !f["parameters"].empty()) {
        const nlohmann::json& params = f["parameters"];
        out += ",parameters:{";
        if (params.contains("properties") && params["properties"].is_object() && !params["properties"].empty()) {
            out += "properties:{";
            gemma4_parameters(params["properties"], false, out);
            out += "},";
        }
        if (params.contains("required") && params["required"].is_array() && !params["required"].empty()) {
            gemma4_required(params["required"], out);
            out += ",";
        }
        if (!gemma4_type(params).empty()) out += "type:<|\"|>" + gemma4_type(params) + "<|\"|>}";
    }
    if (f.contains("response") && f["response"].is_object()) {
        const nlohmann::json& r = f["response"];
        out += ",response:{";
        if (r.contains("description") && r["description"].is_string() && !r["description"].get<std::string>().empty()) {
            out += "description:<|\"|>" + r["description"].get<std::string>() + "<|\"|>,";
        }
        if (gemma4_type(r) == "OBJECT") out += "type:<|\"|>OBJECT<|\"|>}";
    }
    out += "}";
    return std::string(trim(out));
}

// The template's strip_thinking: the <|channel>...<channel|> parts of an answer removed.
std::string gemma4_strip_thinking(std::string_view text) {
    std::string out;
    std::size_t pos = 0;
    while (pos <= text.size()) {
        const std::size_t end = text.find("<channel|>", pos);
        const std::string_view part = text.substr(pos, end == std::string_view::npos ? std::string_view::npos : end - pos);
        const std::size_t open = part.find("<|channel>");
        out += std::string(open == std::string_view::npos ? part : part.substr(0, open));
        if (end == std::string_view::npos) break;
        pos = end + 10;
    }
    return std::string(trim(out));
}

// Gemma 4: <|turn>role\ncontent<turn|>\n per message; the system turn also opens
// thinking (<|think|>) and carries the tool declarations; an assistant's calls and
// the results that follow render inline in its turn, with no model header before
// the continuation. Follows the file's template, which llama.cpp's engine checks.
std::string render_gemma4(const std::vector<ChatMessage>& messages, const ChatOptions& o) {
    std::string out;
    const bool think = o.think && o.think_requested;
    std::size_t first = 0;
    std::string prev_type;  // the template's ns.prev_message_type
    if (think || !o.tools.empty() || (!messages.empty() && messages[0].role == "system")) {
        out += "<|turn>system\n";
        if (think) {
            out += "<|think|>\n";
            prev_type = "think";
        }
        if (!messages.empty() && messages[0].role == "system") {
            out += std::string(trim(messages[0].content));
            first = 1;
        }
        for (const std::string& t : o.tools) out += "<|tool>" + gemma4_declaration(t) + "<tool|>";
        if (!o.tools.empty()) prev_type = "tool";
        out += "<turn|>\n";
    }
    std::size_t last_user = 0;
    bool any_user = false;
    for (std::size_t i = first; i < messages.size(); ++i) {
        if (messages[i].role == "user") {
            last_user = i;
            any_user = true;
        }
    }
    std::string prev_role;  // the previous non-tool message's role
    for (std::size_t i = first; i < messages.size(); ++i) {
        const ChatMessage& m = messages[i];
        check_role(m, true);
        if (m.role == "tool") continue;  // rendered with the call it answers
        if (m.role == "system") throw std::runtime_error("chat: Gemma 4 takes a system message only first");
        prev_type.clear();
        const std::string role = m.role == "assistant" ? "model" : m.role;
        if (!(role == "model" && prev_role == "assistant")) out += "<|turn>" + role + "\n";
        if (!m.reasoning.empty() && any_user && i > last_user) out += "<|channel>thought\n" + m.reasoning + "\n<channel|>";
        for (const ToolCall& c : m.tool_calls) {
            out += "<|tool_call>call:" + c.name + "{";
            const nlohmann::json args = nlohmann::json::parse(c.arguments, nullptr, false);
            if (args.is_object()) {
                bool first_arg = true;
                for (const auto& [key, value] : args.items()) {
                    out += first_arg ? "" : ",";
                    first_arg = false;
                    out += key + ":";
                    gemma4_value(value, false, out);
                }
            } else if (!args.is_null() && !c.arguments.empty()) {
                throw std::runtime_error("chat: Gemma 4 tool-call arguments must be a JSON object");
            }
            out += "}<tool_call|>";
            prev_type = "tool_call";
        }
        bool responded = false;
        if (!m.tool_calls.empty()) {
            for (std::size_t k = i + 1; k < messages.size() && messages[k].role == "tool"; ++k) {
                std::string name = "unknown";
                for (const ToolCall& c : m.tool_calls) {
                    if (!c.id.empty() && c.id == messages[k].tool_call_id) name = c.name;
                }
                out += "<|tool_response>response:" + name + "{value:<|\"|>" + messages[k].content + "<|\"|>}<tool_response|>";
                responded = true;
                prev_type = "tool_response";
            }
        }
        const std::string content = role == "model" ? gemma4_strip_thinking(m.content) : std::string(trim(m.content));
        out += content;
        std::string next_role;
        for (std::size_t j = i + 1; j < messages.size(); ++j) {
            if (messages[j].role != "tool") {
                next_role = messages[j].role;
                break;
            }
        }
        const bool continues = role == "model" && next_role == "assistant" && (m.tool_calls.empty() || responded);
        if (prev_type == "tool_call" && !responded) {
            out += "<|tool_response>";
        } else if (continues) {
        } else if (!(responded && content.empty() && next_role.empty())) {
            out += "<turn|>\n";
        }
        prev_role = m.role;
    }
    if (o.generation_prompt) {
        if (prev_type != "tool_response" && prev_type != "tool_call") {
            out += "<|turn>model\n";
            // The 12B template prefills an empty thought block when thinking is off (the E2B/E4B ones do not).
            if (o.empty_thought && !think) out += "<|channel>thought\n<channel|>";
        } else if (prev_type == "tool_response" && think) {
            out += "<|channel>thought\n";
        }
    }
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
            out += m.content;
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

std::string default_system_of(const GgufFile& file) {
    const std::string_view tmpl = file.get<std::string_view>("tokenizer.chat_template").value_or("");
    // The literal the template emits when messages[0] is not a system message:
    // '<|im_start|>system\n<prompt><|im_end|>' inside quotes, without a '+' (a content expression).
    // The newline is either literal (SmolLM2) or the escape \n (Qwen2.5).
    for (std::string_view open : {std::string_view("<|im_start|>system\n"), std::string_view("<|im_start|>system\\n")}) {
      for (std::size_t at = tmpl.find(open); at != std::string_view::npos; at = tmpl.find(open, at + 1)) {
        const std::size_t start = at + open.size();
        const std::size_t close = tmpl.find("<|im_end|>", start);
        if (close == std::string_view::npos) break;
        const std::string_view body = tmpl.substr(start, close - start);
        if (body.empty() || body.find('\'') != std::string_view::npos || body.find('"') != std::string_view::npos ||
            body.find('{') != std::string_view::npos) {
            continue;
        }
        return std::string(body);
      }
    }
    return "";
}

bool chat_template_has_tools(const GgufFile& file) {
    const std::string_view tmpl = file.get<std::string_view>("tokenizer.chat_template").value_or("");
    return chat_format_has_tools(chat_format_of(file)) && tmpl.find("tools") != std::string_view::npos;
}

std::string render_chat(const GgufFile& file, const std::vector<ChatMessage>& messages, ChatOptions options) {
    const ChatFormat format = chat_format_of(file);
    if (format == ChatFormat::ChatML) options.default_system = default_system_of(file);
    if (format == ChatFormat::Gemma4) {
        const std::string_view tmpl = file.get<std::string_view>("tokenizer.chat_template").value_or("");
        options.empty_thought = tmpl.find("'<|channel>thought\\n<channel|>'") != std::string_view::npos;
    }
    if (!options.tools.empty() && !chat_template_has_tools(file)) {
        throw std::runtime_error("chat: this model's template has no tool format (Qwen, Llama 3.x and Gemma 4 files take tools)");
    }
    return render_chat(format, messages, options);
}

namespace {

// A recursive-descent parse of Gemma 4's argument syntax into JSON.
struct Gemma4Parser {
    std::string_view s;
    std::size_t pos = 0;
    bool ok = true;

    void skip() {
        while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) ++pos;
    }
    bool take(std::string_view lit) {
        skip();
        if (s.substr(pos, lit.size()) != lit) return false;
        pos += lit.size();
        return true;
    }
    ojson value() {
        skip();
        if (take("<|\"|>")) {
            const std::size_t end = s.find("<|\"|>", pos);
            if (end == std::string_view::npos) return fail();
            const std::string text(s.substr(pos, end - pos));
            pos = end + 5;
            return text;
        }
        if (take("null")) return nullptr;
        if (take("true")) return true;
        if (take("false")) return false;
        if (take("{")) {
            ojson obj = ojson::object();
            if (take("}")) return obj;
            do {
                skip();
                std::size_t start = pos;
                if (take("<|\"|>")) {  // an escaped key
                    const std::size_t end = s.find("<|\"|>", pos);
                    if (end == std::string_view::npos) return fail();
                    start = pos;
                    pos = end;
                    const std::string key(s.substr(start, end - start));
                    pos = end + 5;
                    if (!take(":")) return fail();
                    obj[key] = value();
                } else {
                    while (pos < s.size() && s[pos] != ':' && s[pos] != '}' && s[pos] != ',') ++pos;
                    const std::string key(trim(s.substr(start, pos - start)));
                    if (key.empty() || !take(":")) return fail();
                    obj[key] = value();
                }
                if (!ok) return obj;
            } while (take(","));
            if (!take("}")) return fail();
            return obj;
        }
        if (take("[")) {
            ojson arr = ojson::array();
            if (take("]")) return arr;
            do {
                arr.push_back(value());
                if (!ok) return arr;
            } while (take(","));
            if (!take("]")) return fail();
            return arr;
        }
        // a number
        const std::size_t start = pos;
        while (pos < s.size() && (std::isalnum(static_cast<unsigned char>(s[pos])) || s[pos] == '-' || s[pos] == '+' || s[pos] == '.')) ++pos;
        const ojson n = ojson::parse(s.substr(start, pos - start), nullptr, false);
        if (n.is_discarded() || !n.is_number()) return fail();
        return n;
    }
    ojson fail() {
        ok = false;
        return nullptr;
    }
};

}  // namespace

std::string gemma4_arguments_to_json(std::string_view text) {
    Gemma4Parser p{text};
    const ojson v = p.value();
    p.skip();
    if (!p.ok || !v.is_object() || p.pos != text.size()) return "";
    return v.dump();
}

bool chat_format_has_tools(ChatFormat format) {
    return format == ChatFormat::ChatML || format == ChatFormat::Qwen3 || format == ChatFormat::Llama3 || format == ChatFormat::Gemma4;
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
