#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace brisk {

// Metadata value types, numbered as in the GGUF specification.
enum class GgufType : std::uint32_t {
    U8 = 0, I8 = 1, U16 = 2, I16 = 3, U32 = 4, I32 = 5, F32 = 6, Bool = 7,
    String = 8, Array = 9, U64 = 10, I64 = 11, F64 = 12,
};

// Tensor storage types, numbered as in ggml. Only the ones brisk can size are named.
enum class TensorType : std::uint32_t {
    F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q8_1 = 9,
    Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, Q8_K = 15,
    I8 = 24, I16 = 25, I32 = 26, I64 = 27, F64 = 28, BF16 = 30,
    TQ1_0 = 34, TQ2_0 = 35,
};

std::string_view to_string(GgufType type);
// Returns "unknown" for types brisk does not recognise.
std::string_view to_string(TensorType type);

// An array value, left undecoded in the file. Token vocabularies have over
// 100k entries, so decoding is done on request rather than at parse time.
struct GgufArray {
    GgufType elem_type{};
    std::uint64_t count = 0;
    std::span<const std::byte> data;  // the encoded elements, already bounds-checked

    // Throws std::runtime_error if elem_type is not String.
    std::vector<std::string_view> strings() const;
    // Throws std::runtime_error unless T matches elem_type exactly.
    template <typename T>
    std::vector<T> values() const;
};

using GgufValue = std::variant<std::uint8_t, std::int8_t, std::uint16_t, std::int16_t,
                               std::uint32_t, std::int32_t, float, bool, std::string_view,
                               GgufArray, std::uint64_t, std::int64_t, double>;

struct TensorInfo {
    std::string_view name;
    std::vector<std::uint64_t> dims;  // ggml order: innermost dimension first
    TensorType type{};
    std::uint64_t offset = 0;         // relative to the start of the tensor data section
    std::span<const std::byte> data;  // empty when the type is unknown to brisk

    std::uint64_t n_elements() const;
};

// A parsed GGUF file. All string_views and spans point into the buffer passed
// to parse(), which must outlive this object.
class GgufFile {
public:
    // Throws std::runtime_error on malformed or truncated input; never reads
    // outside `bytes`.
    static GgufFile parse(std::span<const std::byte> bytes);

    std::uint32_t version() const { return version_; }
    const std::vector<std::pair<std::string_view, GgufValue>>& metadata() const { return metadata_; }
    const std::vector<TensorInfo>& tensors() const { return tensors_; }

    const GgufValue* find(std::string_view key) const;
    const TensorInfo* tensor(std::string_view name) const;

    // Returns the value if the key exists and holds exactly type T.
    template <typename T>
    std::optional<T> get(std::string_view key) const {
        const GgufValue* value = find(key);
        if (value == nullptr) return std::nullopt;
        if (const T* typed = std::get_if<T>(value)) return *typed;
        return std::nullopt;
    }

private:
    std::uint32_t version_ = 0;
    std::vector<std::pair<std::string_view, GgufValue>> metadata_;
    std::vector<TensorInfo> tensors_;
    std::unordered_map<std::string_view, std::size_t> metadata_index_;
    std::unordered_map<std::string_view, std::size_t> tensor_index_;
};

}  // namespace brisk
