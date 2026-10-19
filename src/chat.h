#pragma once
// Chat prompts: a conversation rendered in a model family's turn format.
#include <string>
#include <vector>

#include "model.h"

namespace brisk {

struct ChatMessage {
    std::string role;  // "system", "user" or "assistant"
    std::string content;
};

struct ChatOptions {
    bool think = true;           // Qwen3: leave the reasoning block to the model (false: an empty one is prefilled)
    bool generation_prompt = true;  // end with the assistant's turn header
};

// The conversation in the family's format, without the BOS token (the
// tokenizer adds that). Each family's rendering follows the chat_template of
// its GGUF files, restricted to text messages: Qwen's ChatML, Llama 3's
// headers, Gemma's turns (the system message folded into the first user
// turn for Gemma 3, its own turn for Gemma 4), BitNet's "Role: " lines.
std::string render_chat(Architecture architecture, const std::vector<ChatMessage>& messages, const ChatOptions& options = {});

}  // namespace brisk
