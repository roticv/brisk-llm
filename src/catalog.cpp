#include "catalog.h"

#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace brisk {

namespace {

constexpr std::string_view kHf = "https://huggingface.co/";

// Verified files: each tokenizes identically to llama.cpp on the test corpus,
// matches its perplexity, renders the chat template it does, and is pinned in
// tests/perplexity_pins.json. Sizes are the published files' (2026-10-24).
constexpr CatalogEntry kCatalog[] = {
    {"qwen3-0.6b", "Qwen3-0.6B-Q4_0.gguf", "unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_0.gguf", 382156480,
     "Qwen3 0.6B, Q4_0 (Q6_K embedding): the smallest; the CI's file"},
    {"qwen3-1.7b", "Qwen3-1.7B-Q4_0.gguf", "unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_0.gguf", 1056782912,
     "Qwen3 1.7B, Q4_0"},
    {"qwen3-1.7b-k", "Qwen3-1.7B-Q4_K_M.gguf", "unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf", 1107409472,
     "Qwen3 1.7B, Q4_K_M"},
    {"qwen2.5-1.5b", "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
     "bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf", 986048768, "Qwen2.5 1.5B Instruct, Q4_K_M"},
    {"deepseek-r1-1.5b", "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
     "bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf", 1117320800,
     "DeepSeek-R1 distilled into Qwen2.5 1.5B, Q4_K_M (a reasoning model: it thinks before answering)"},
    {"llama-3.2-1b", "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
     "bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf", 807694464, "Llama 3.2 1B Instruct, Q4_K_M"},
    {"llama-3.2-3b", "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
     "bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf", 2019377696, "Llama 3.2 3B Instruct, Q4_K_M"},
    {"mistral-7b", "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
     "bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf", 4372812000,
     "Mistral 7B Instruct v0.3, Q4_K_M (4.4 GB: not for a 4 GB Pi)"},
    {"smollm2-1.7b", "SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
     "bartowski/SmolLM2-1.7B-Instruct-GGUF/resolve/main/SmolLM2-1.7B-Instruct-Q4_K_M.gguf", 1055609824, "SmolLM2 1.7B Instruct, Q4_K_M"},
    {"gemma-3-1b", "gemma-3-1b-it-Q4_K_M.gguf", "ggml-org/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf", 806058240,
     "Gemma 3 1B instruction-tuned, Q4_K_M"},
    {"gemma-4-e2b", "gemma-4-E2B-it-Q4_0.gguf", "ggml-org/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_0.gguf", 2841481184,
     "Gemma 4 E2B instruction-tuned, Q4_0 (2.8 GB: fits a 4 GB Pi with --ctx 1024)"},
    {"gemma-4-e4b", "gemma-4-E4B-it-Q4_0.gguf", "ggml-org/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_0.gguf", 4590807392,
     "Gemma 4 E4B instruction-tuned, Q4_0 (4.6 GB)"},
    {"gemma-4-12b", "gemma-4-12B-it-Q4_0.gguf", "ggml-org/gemma-4-12B-it-GGUF/resolve/main/gemma-4-12B-it-Q4_0.gguf", 7219673216,
     "Gemma 4 12B instruction-tuned, Q4_0 (7.2 GB; the Mac's GPU)"},
    {"phi-4-mini", "Phi-4-mini-instruct-Q4_K_M.gguf",
     "bartowski/microsoft_Phi-4-mini-instruct-GGUF/resolve/main/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf", 2491874688,
     "Phi-4-mini 3.8B instruct, Q4_K_M (2.5 GB)"},
    {"lfm2-1.2b", "LFM2-1.2B-Q4_0.gguf", "LiquidAI/LFM2-1.2B-GGUF/resolve/main/LFM2-1.2B-Q4_0.gguf", 695749568,
     "LFM2 1.2B, Q4_0: Liquid's edge model, the pick for the Pi"},
};

}  // namespace

std::span<const CatalogEntry> catalog() { return kCatalog; }

const CatalogEntry* catalog_find(std::string_view name) {
    for (const CatalogEntry& e : kCatalog) {
        if (e.name == name || e.file == name) return &e;
    }
    return nullptr;
}

std::string pull(const CatalogEntry& entry, const std::string& dir, bool quiet) {
    namespace fs = std::filesystem;
    fs::create_directories(dir);
    const fs::path target = fs::path(dir) / std::string(entry.file);
    if (fs::exists(target) && fs::file_size(target) == entry.bytes) return target.string();
    const fs::path part = target.string() + ".part";
    // The system's curl: follows the redirect to the CDN and resumes a partial file.
    const std::string command = std::string("curl -fL ") + (quiet ? "-sS " : "--progress-bar ") + "-C - -o '" + part.string() +
                                "' '" + std::string(kHf) + std::string(entry.url) + "'";
    const int status = std::system(command.c_str());
    if (status != 0) throw std::runtime_error("download failed (curl exit " + std::to_string(status) + "); the partial file is kept for a retry");
    if (!fs::exists(part) || fs::file_size(part) != entry.bytes) {
        throw std::runtime_error("downloaded " + std::to_string(fs::exists(part) ? fs::file_size(part) : 0) + " bytes, expected " +
                                 std::to_string(entry.bytes) + "; the published file may have changed");
    }
    fs::rename(part, target);
    return target.string();
}

}  // namespace brisk
