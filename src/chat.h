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
    bool generation_prompt = true;  // end with the assistant's turn header
    std::vector<std::string> tools;  // tool definitions as JSON text (OpenAI's {"type": "function", "function": ...})
};

// Whether a format renders tools, tool calls and tool results (ChatML and Llama 3 do).
bool chat_format_has_tools(ChatFormat format);

// The conversation in the family's format, without the BOS token (the
// tokenizer adds that). Each family's rendering follows the chat_template of
// its GGUF files, restricted to text messages: Qwen's ChatML, Llama 3's
// headers, Gemma's turns (the system message folded into the first user
// turn for Gemma 3, its own turn for Gemma 4), Mistral's [INST], DeepSeek
// R1's <｜User｜> markers, BitNet's "Role: " lines. Where llama.cpp's built-in
// renderer knows the template, the output is identical to it.
std::string render_chat(ChatFormat format, const std::vector<ChatMessage>& messages, const ChatOptions& options = {});

}  // namespace brisk
