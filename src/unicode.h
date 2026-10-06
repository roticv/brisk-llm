#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace brisk {

// A decoded UTF-8 sequence. Invalid bytes decode one at a time with
// `valid == false`, so the original bytes can always be recovered from
// `offset` and `length`.
struct Codepoint {
    std::uint32_t value = 0;
    std::uint32_t offset = 0;  // byte offset into the source text
    std::uint8_t length = 0;   // bytes consumed, 1 to 4
    bool valid = false;
};

std::vector<Codepoint> decode_utf8(std::string_view text);
void append_utf8(std::uint32_t codepoint, std::string& out);

bool is_letter(std::uint32_t codepoint);      // general category L*
bool is_number(std::uint32_t codepoint);      // general category N*
bool is_whitespace(std::uint32_t codepoint);  // Unicode White_Space property

}  // namespace brisk
