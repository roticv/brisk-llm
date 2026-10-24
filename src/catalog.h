#pragma once
// The model files brisk is checked against (tokenizer, perplexity and chat
// template against llama.cpp; pinned), by short name, for `brisk pull`.
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace brisk {

struct CatalogEntry {
    std::string_view name;   // what `brisk pull` takes
    std::string_view file;   // the file name under models/
    std::string_view url;    // where it comes from
    std::uint64_t bytes;     // its size, checked after the download
    std::string_view note;   // family and what the file is for
};

std::span<const CatalogEntry> catalog();
const CatalogEntry* catalog_find(std::string_view name);

// Downloads an entry into `dir` (resuming a partial file), verifies the size
// and returns the path. Uses the system's curl. Throws std::runtime_error.
std::string pull(const CatalogEntry& entry, const std::string& dir, bool quiet);

}  // namespace brisk
