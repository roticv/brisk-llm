#include "gguf.h"

#include <bit>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace brisk {

static_assert(std::endian::native == std::endian::little, "GGUF is little-endian; so must the host be");

namespace {

constexpr std::uint32_t kMagic = 0x46554747;  // "GGUF"
constexpr std::uint32_t kDefaultAlignment = 32;
constexpr std::uint32_t kMaxDims = 4;
// Smallest possible encodings, used to reject absurd counts before reserving.
constexpr std::size_t kMinKeyValueBytes = 8 + 4;
constexpr std::size_t kMinTensorInfoBytes = 8 + 4 + 4 + 8;

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("gguf: " + what); }

// Bounds-checked cursor over the file bytes.
class Reader {
public:
    explicit Reader(std::span<const std::byte> buf) : buf_(buf) {}

    std::size_t pos() const { return pos_; }
    std::size_t remaining() const { return buf_.size() - pos_; }

    std::span<const std::byte> take(std::uint64_t n) {
        if (n > remaining()) fail("unexpected end of file");
        const auto out = buf_.subspan(pos_, static_cast<std::size_t>(n));
        pos_ += static_cast<std::size_t>(n);
        return out;
    }

    template <typename T>
    T read() {
        static_assert(std::is_trivially_copyable_v<T>);
        T value;
        std::memcpy(&value, take(sizeof(T)).data(), sizeof(T));
        return value;
    }

    std::string_view read_string() {
        const auto len = read<std::uint64_t>();
        const auto bytes = take(len);
        return {reinterpret_cast<const char*>(bytes.data()), bytes.size()};
    }

private:
    std::span<const std::byte> buf_;
    std::size_t pos_ = 0;
};

// Byte size of a fixed-width metadata type; 0 for String and Array.
std::size_t scalar_size(GgufType type) {
    switch (type) {
        case GgufType::U8:
        case GgufType::I8:
        case GgufType::Bool:
            return 1;
        case GgufType::U16:
        case GgufType::I16:
            return 2;
        case GgufType::U32:
        case GgufType::I32:
        case GgufType::F32:
            return 4;
        case GgufType::U64:
        case GgufType::I64:
        case GgufType::F64:
            return 8;
        case GgufType::String:
        case GgufType::Array:
            return 0;
    }
    return 0;
}

GgufType read_type(Reader& r) {
    const auto raw = r.read<std::uint32_t>();
    if (raw > static_cast<std::uint32_t>(GgufType::F64)) fail("unknown metadata type " + std::to_string(raw));
    return static_cast<GgufType>(raw);
}

GgufArray read_array(Reader& r) {
    GgufArray array;
    array.elem_type = read_type(r);
    array.count = r.read<std::uint64_t>();

    if (array.elem_type == GgufType::Array) fail("nested arrays are not supported");
    if (array.elem_type == GgufType::String) {
        const std::size_t start = r.pos();
        Reader scan = r;
        for (std::uint64_t i = 0; i < array.count; ++i) scan.read_string();
        array.data = r.take(scan.pos() - start);
        return array;
    }

    std::uint64_t bytes = 0;
    if (__builtin_mul_overflow(array.count, static_cast<std::uint64_t>(scalar_size(array.elem_type)), &bytes)) {
        fail("array size overflows");
    }
    array.data = r.take(bytes);
    return array;
}

GgufValue read_value(Reader& r, GgufType type) {
    switch (type) {
        case GgufType::U8: return r.read<std::uint8_t>();
        case GgufType::I8: return r.read<std::int8_t>();
        case GgufType::U16: return r.read<std::uint16_t>();
        case GgufType::I16: return r.read<std::int16_t>();
        case GgufType::U32: return r.read<std::uint32_t>();
        case GgufType::I32: return r.read<std::int32_t>();
        case GgufType::F32: return r.read<float>();
        case GgufType::Bool: return r.read<std::uint8_t>() != 0;
        case GgufType::String: return r.read_string();
        case GgufType::Array: return read_array(r);
        case GgufType::U64: return r.read<std::uint64_t>();
        case GgufType::I64: return r.read<std::int64_t>();
        case GgufType::F64: return r.read<double>();
    }
    fail("unknown metadata type");
}

struct BlockLayout {
    std::uint64_t elements;
    std::uint64_t bytes;
};

// Elements and bytes per storage block, matching ggml's type traits.
std::optional<BlockLayout> block_layout(TensorType type) {
    switch (type) {
        case TensorType::F32: return BlockLayout{1, 4};
        case TensorType::F16: return BlockLayout{1, 2};
        case TensorType::BF16: return BlockLayout{1, 2};
        case TensorType::F64: return BlockLayout{1, 8};
        case TensorType::I8: return BlockLayout{1, 1};
        case TensorType::I16: return BlockLayout{1, 2};
        case TensorType::I32: return BlockLayout{1, 4};
        case TensorType::I64: return BlockLayout{1, 8};
        case TensorType::Q4_0: return BlockLayout{32, 18};
        case TensorType::Q4_1: return BlockLayout{32, 20};
        case TensorType::Q5_0: return BlockLayout{32, 22};
        case TensorType::Q5_1: return BlockLayout{32, 24};
        case TensorType::Q8_0: return BlockLayout{32, 34};
        case TensorType::Q8_1: return BlockLayout{32, 36};
        case TensorType::Q2_K: return BlockLayout{256, 84};
        case TensorType::Q3_K: return BlockLayout{256, 110};
        case TensorType::Q4_K: return BlockLayout{256, 144};
        case TensorType::Q5_K: return BlockLayout{256, 176};
        case TensorType::Q6_K: return BlockLayout{256, 210};
        case TensorType::Q8_K: return BlockLayout{256, 292};
        case TensorType::TQ1_0: return BlockLayout{256, 54};
        case TensorType::TQ2_0: return BlockLayout{256, 66};
    }
    return std::nullopt;
}

template <typename T>
constexpr GgufType gguf_type_of() {
    if constexpr (std::is_same_v<T, std::uint8_t>) return GgufType::U8;
    else if constexpr (std::is_same_v<T, std::int8_t>) return GgufType::I8;
    else if constexpr (std::is_same_v<T, std::uint16_t>) return GgufType::U16;
    else if constexpr (std::is_same_v<T, std::int16_t>) return GgufType::I16;
    else if constexpr (std::is_same_v<T, std::uint32_t>) return GgufType::U32;
    else if constexpr (std::is_same_v<T, std::int32_t>) return GgufType::I32;
    else if constexpr (std::is_same_v<T, float>) return GgufType::F32;
    else if constexpr (std::is_same_v<T, std::uint64_t>) return GgufType::U64;
    else if constexpr (std::is_same_v<T, std::int64_t>) return GgufType::I64;
    else if constexpr (std::is_same_v<T, double>) return GgufType::F64;
    else static_assert(sizeof(T) == 0, "unsupported array element type");
}

}  // namespace

std::string_view to_string(GgufType type) {
    switch (type) {
        case GgufType::U8: return "u8";
        case GgufType::I8: return "i8";
        case GgufType::U16: return "u16";
        case GgufType::I16: return "i16";
        case GgufType::U32: return "u32";
        case GgufType::I32: return "i32";
        case GgufType::F32: return "f32";
        case GgufType::Bool: return "bool";
        case GgufType::String: return "string";
        case GgufType::Array: return "array";
        case GgufType::U64: return "u64";
        case GgufType::I64: return "i64";
        case GgufType::F64: return "f64";
    }
    return "unknown";
}

std::string_view to_string(TensorType type) {
    switch (type) {
        case TensorType::F32: return "F32";
        case TensorType::F16: return "F16";
        case TensorType::BF16: return "BF16";
        case TensorType::F64: return "F64";
        case TensorType::I8: return "I8";
        case TensorType::I16: return "I16";
        case TensorType::I32: return "I32";
        case TensorType::I64: return "I64";
        case TensorType::Q4_0: return "Q4_0";
        case TensorType::Q4_1: return "Q4_1";
        case TensorType::Q5_0: return "Q5_0";
        case TensorType::Q5_1: return "Q5_1";
        case TensorType::Q8_0: return "Q8_0";
        case TensorType::Q8_1: return "Q8_1";
        case TensorType::Q2_K: return "Q2_K";
        case TensorType::Q3_K: return "Q3_K";
        case TensorType::Q4_K: return "Q4_K";
        case TensorType::Q5_K: return "Q5_K";
        case TensorType::Q6_K: return "Q6_K";
        case TensorType::Q8_K: return "Q8_K";
        case TensorType::TQ1_0: return "TQ1_0";
        case TensorType::TQ2_0: return "TQ2_0";
    }
    return "unknown";
}

std::vector<std::string_view> GgufArray::strings() const {
    if (elem_type != GgufType::String) fail("array does not hold strings");
    std::vector<std::string_view> out;
    out.reserve(static_cast<std::size_t>(count));
    Reader r(data);
    for (std::uint64_t i = 0; i < count; ++i) out.push_back(r.read_string());
    return out;
}

template <typename T>
std::vector<T> GgufArray::values() const {
    if (elem_type != gguf_type_of<T>()) fail("array element type mismatch");
    std::vector<T> out(static_cast<std::size_t>(count));
    if (!out.empty()) std::memcpy(out.data(), data.data(), out.size() * sizeof(T));
    return out;
}

template std::vector<std::uint8_t> GgufArray::values() const;
template std::vector<std::int8_t> GgufArray::values() const;
template std::vector<std::uint16_t> GgufArray::values() const;
template std::vector<std::int16_t> GgufArray::values() const;
template std::vector<std::uint32_t> GgufArray::values() const;
template std::vector<std::int32_t> GgufArray::values() const;
template std::vector<float> GgufArray::values() const;
template std::vector<std::uint64_t> GgufArray::values() const;
template std::vector<std::int64_t> GgufArray::values() const;
template std::vector<double> GgufArray::values() const;

std::uint64_t TensorInfo::n_elements() const {
    std::uint64_t n = 1;
    for (const std::uint64_t d : dims) n *= d;  // parse() already rejected overflow
    return n;
}

const GgufValue* GgufFile::find(std::string_view key) const {
    const auto it = metadata_index_.find(key);
    return it == metadata_index_.end() ? nullptr : &metadata_[it->second].second;
}

const TensorInfo* GgufFile::tensor(std::string_view name) const {
    const auto it = tensor_index_.find(name);
    return it == tensor_index_.end() ? nullptr : &tensors_[it->second];
}

GgufFile GgufFile::parse(std::span<const std::byte> bytes) {
    Reader r(bytes);
    if (r.read<std::uint32_t>() != kMagic) fail("bad magic, not a GGUF file");

    GgufFile file;
    file.version_ = r.read<std::uint32_t>();
    if (file.version_ != 2 && file.version_ != 3) fail("unsupported version " + std::to_string(file.version_));

    const auto tensor_count = r.read<std::uint64_t>();
    const auto kv_count = r.read<std::uint64_t>();
    if (kv_count > r.remaining() / kMinKeyValueBytes) fail("metadata count exceeds file size");
    if (tensor_count > r.remaining() / kMinTensorInfoBytes) fail("tensor count exceeds file size");

    file.metadata_.reserve(static_cast<std::size_t>(kv_count));
    for (std::uint64_t i = 0; i < kv_count; ++i) {
        const std::string_view key = r.read_string();
        const GgufType type = read_type(r);
        GgufValue value = read_value(r, type);
        if (!file.metadata_index_.emplace(key, file.metadata_.size()).second) {
            fail("duplicate metadata key '" + std::string(key) + "'");
        }
        file.metadata_.emplace_back(key, std::move(value));
    }

    file.tensors_.reserve(static_cast<std::size_t>(tensor_count));
    for (std::uint64_t i = 0; i < tensor_count; ++i) {
        TensorInfo info;
        info.name = r.read_string();
        const auto n_dims = r.read<std::uint32_t>();
        if (n_dims > kMaxDims) fail("tensor '" + std::string(info.name) + "' has too many dimensions");
        std::uint64_t elements = 1;
        for (std::uint32_t d = 0; d < n_dims; ++d) {
            const auto dim = r.read<std::uint64_t>();
            if (__builtin_mul_overflow(elements, dim, &elements)) {
                fail("tensor '" + std::string(info.name) + "' element count overflows");
            }
            info.dims.push_back(dim);
        }
        info.type = static_cast<TensorType>(r.read<std::uint32_t>());
        info.offset = r.read<std::uint64_t>();
        if (!file.tensor_index_.emplace(info.name, file.tensors_.size()).second) {
            fail("duplicate tensor name '" + std::string(info.name) + "'");
        }
        file.tensors_.push_back(std::move(info));
    }

    const std::uint32_t alignment = file.get<std::uint32_t>("general.alignment").value_or(kDefaultAlignment);
    if (!std::has_single_bit(alignment)) fail("alignment is not a power of two");

    const std::size_t padding = (alignment - r.pos() % alignment) % alignment;
    r.take(padding);
    const std::span<const std::byte> data = bytes.subspan(r.pos());

    for (TensorInfo& info : file.tensors_) {
        const auto layout = block_layout(info.type);
        if (!layout) continue;  // unknown type: keep the entry, leave data empty

        const std::uint64_t elements = info.n_elements();
        if (elements % layout->elements != 0) {
            fail("tensor '" + std::string(info.name) + "' size is not a whole number of blocks");
        }
        std::uint64_t size = 0;
        if (__builtin_mul_overflow(elements / layout->elements, layout->bytes, &size)) {
            fail("tensor '" + std::string(info.name) + "' byte size overflows");
        }
        if (info.offset > data.size() || size > data.size() - info.offset) {
            fail("tensor '" + std::string(info.name) + "' data lies outside the file");
        }
        info.data = data.subspan(static_cast<std::size_t>(info.offset), static_cast<std::size_t>(size));
    }

    return file;
}

}  // namespace brisk
