#pragma once
// Chat prompts: a conversation rendered in a model family's turn format.
#include <string>
#include <vector>

#include "gguf.h"
#include "model.h"

namespace brisk {

// A function call an assistant message made (OpenAI's tool_calls).
struct ToolCall {
    std::string name;
    std::string arguments;  // JSON text
};

struct ChatMessage {
    std::string role;  // "system", "user", "assistant" or "tool" (a call's result)
    std::string content;
    std::vector<ToolCall> tool_calls;  // assistant only
};

// A turn format. Chosen from the file's chat_template (several families
// share one: SmolLM2 is a Llama that speaks ChatML), with the architecture
// as the fallback.
enum class ChatFormat { ChatML, Qwen3, Llama3, Gemma3, Gemma4, Mistral, DeepSeek, Phi, BitNet };

ChatFormat chat_format_of(const GgufFile& file);

struct ChatOptions {
    bool think = true;           // Qwen3: leave the reasoning block to the model (false: an empty one is prefilled)
    bool think_requested = false;  // Gemma 4: thinking is off in its template unless asked for (<|think|> in the system turn)
    bool generation_prompt = true;  // end with the assistant's turn header
    std::vector<std::string> tools;  // tool definitions as JSON text (OpenAI's {"type": "function", "function": ...})
    std::string default_system;      // ChatML: the system prompt the template supplies when the conversation has none
    bool empty_thought = false;      // Gemma 4 12B: the template prefills an empty thought block (thinking off)
};

// The system prompt a ChatML template inserts when the conversation has none
// (Qwen2.5, SmolLM2), or empty.
std::string default_system_of(const GgufFile& file);

// Whether a format renders tools, tool calls and tool results (ChatML and Llama 3 do),
// and whether a file's own template takes them (SmolLM2's ChatML template does not).
bool chat_format_has_tools(ChatFormat format);
bool chat_template_has_tools(const GgufFile& file);

// The conversation in the family's format, without the BOS token (the
// tokenizer adds that). Each family's rendering follows the chat_template of
// its GGUF files, restricted to text messages: Qwen's ChatML, Llama 3's
// headers, Gemma's turns (the system message folded into the first user
// turn for Gemma 3, its own turn for Gemma 4), Mistral's [INST], DeepSeek
// R1's <｜User｜> markers, BitNet's "Role: " lines. Where llama.cpp's built-in
// renderer knows the template, the output is identical to it.
std::string render_chat(ChatFormat format, const std::vector<ChatMessage>& messages, const ChatOptions& options = {});

// The same for a file: its format and its default system prompt.
std::string render_chat(const GgufFile& file, const std::vector<ChatMessage>& messages, ChatOptions options = {});

}  // namespace brisk
