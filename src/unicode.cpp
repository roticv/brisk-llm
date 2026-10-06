#include "unicode.h"

#include <algorithm>
#include <iterator>

namespace brisk {

namespace {

struct CodepointRange {
    std::uint32_t first;
    std::uint32_t last;
};

#include "unicode_tables.inc"

template <std::size_t N>
bool in_ranges(const CodepointRange (&table)[N], std::uint32_t cp) {
    const auto* it = std::upper_bound(std::begin(table), std::end(table), cp,
                                      [](std::uint32_t value, const CodepointRange& r) { return value < r.first; });
    return it != std::begin(table) && cp <= std::prev(it)->last;
}

bool is_continuation(unsigned char byte) { return (byte & 0xC0) == 0x80; }

}  // namespace

std::vector<Codepoint> decode_utf8(std::string_view text) {
    std::vector<Codepoint> out;
    out.reserve(text.size());

    std::size_t i = 0;
    while (i < text.size()) {
        const auto b0 = static_cast<unsigned char>(text[i]);
        std::uint32_t value = b0;
        std::uint8_t length = 1;
        std::uint32_t min_value = 0;

        if (b0 < 0x80) {
            out.push_back({value, static_cast<std::uint32_t>(i), 1, true});
            ++i;
            continue;
        }
        if ((b0 & 0xE0) == 0xC0) {
            length = 2;
            value = b0 & 0x1F;
            min_value = 0x80;
        } else if ((b0 & 0xF0) == 0xE0) {
            length = 3;
            value = b0 & 0x0F;
            min_value = 0x800;
        } else if ((b0 & 0xF8) == 0xF0) {
            length = 4;
            value = b0 & 0x07;
            min_value = 0x10000;
        }

        bool valid = length > 1 && i + length <= text.size();
        for (std::size_t k = 1; valid && k < length; ++k) {
            const auto b = static_cast<unsigned char>(text[i + k]);
            valid = is_continuation(b);
            value = (value << 6) | (b & 0x3F);
        }
        // Reject overlong encodings, surrogates and values beyond U+10FFFF.
        valid = valid && value >= min_value && value <= 0x10FFFF && !(value >= 0xD800 && value <= 0xDFFF);

        if (valid) {
            out.push_back({value, static_cast<std::uint32_t>(i), length, true});
            i += length;
        } else {
            out.push_back({b0, static_cast<std::uint32_t>(i), 1, false});
            ++i;
        }
    }
    return out;
}

void append_utf8(std::uint32_t cp, std::string& out) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

bool is_letter(std::uint32_t cp) {
    if (cp < 0x80) return (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z');
    return in_ranges(kLetterRanges, cp);
}

bool is_number(std::uint32_t cp) {
    if (cp < 0x80) return cp >= '0' && cp <= '9';
    return in_ranges(kNumberRanges, cp);
}

bool is_whitespace(std::uint32_t cp) {
    return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F ||
           cp == 0x3000;
}

}  // namespace brisk
