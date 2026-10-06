#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "gguf.h"

namespace {

using brisk::GgufFile;
using brisk::GgufType;
using brisk::TensorType;

#define CHECK(cond)                                                             \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

// Builds GGUF bytes by hand so the parser is tested against the format, not
// against another implementation's writer.
class Builder {
public:
    template <typename T>
    Builder& put(T value) {
        const auto* p = reinterpret_cast<const std::byte*>(&value);
        bytes_.insert(bytes_.end(), p, p + sizeof(T));
        return *this;
    }
    Builder& str(std::string_view s) {
        put<std::uint64_t>(s.size());
        const auto* p = reinterpret_cast<const std::byte*>(s.data());
        bytes_.insert(bytes_.end(), p, p + s.size());
        return *this;
    }
    Builder& type(GgufType t) { return put(static_cast<std::uint32_t>(t)); }
    Builder& pad_to(std::size_t alignment) {
        while (bytes_.size() % alignment != 0) bytes_.push_back(std::byte{0});
        return *this;
    }
    std::vector<std::byte>& bytes() { return bytes_; }

private:
    std::vector<std::byte> bytes_;
};

Builder header(std::uint64_t tensors, std::uint64_t kvs) {
    Builder b;
    b.put<std::uint32_t>(0x46554747).put<std::uint32_t>(3).put(tensors).put(kvs);
    return b;
}

std::vector<std::byte> sample_file() {
    Builder b = header(2, 4);
    b.str("general.architecture").type(GgufType::String).str("qwen3");
    b.str("qwen3.block_count").type(GgufType::U32).put<std::uint32_t>(28);
    b.str("tokenizer.ggml.tokens").type(GgufType::Array).type(GgufType::String).put<std::uint64_t>(3);
    b.str("a").str("").str("hello");
    b.str("tokenizer.ggml.scores").type(GgufType::Array).type(GgufType::F32).put<std::uint64_t>(2);
    b.put(1.5f).put(-2.0f);

    b.str("w.f32").put<std::uint32_t>(2).put<std::uint64_t>(3).put<std::uint64_t>(2);
    b.put(static_cast<std::uint32_t>(TensorType::F32)).put<std::uint64_t>(0);
    b.str("w.q4_0").put<std::uint32_t>(2).put<std::uint64_t>(32).put<std::uint64_t>(2);
    b.put(static_cast<std::uint32_t>(TensorType::Q4_0)).put<std::uint64_t>(32);

    b.pad_to(32);
    for (int i = 0; i < 6; ++i) b.put(static_cast<float>(i));
    b.pad_to(32);
    for (int i = 0; i < 36; ++i) b.put(static_cast<std::uint8_t>(i));
    return b.bytes();
}

bool throws(const std::vector<std::byte>& bytes) {
    try {
        GgufFile::parse(bytes);
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

void test_parses_valid_file() {
    const std::vector<std::byte> bytes = sample_file();
    const GgufFile file = GgufFile::parse(bytes);

    CHECK(file.version() == 3);
    CHECK(file.metadata().size() == 4);
    CHECK(file.get<std::string_view>("general.architecture") == "qwen3");
    CHECK(file.get<std::uint32_t>("qwen3.block_count") == 28u);
    CHECK(!file.get<std::int32_t>("qwen3.block_count"));  // wrong type
    CHECK(!file.get<std::uint32_t>("missing"));

    const auto tokens = file.get<brisk::GgufArray>("tokenizer.ggml.tokens");
    CHECK(tokens && tokens->count == 3);
    const auto strings = tokens->strings();
    CHECK(strings.size() == 3 && strings[0] == "a" && strings[1].empty() && strings[2] == "hello");

    const auto scores = file.get<brisk::GgufArray>("tokenizer.ggml.scores");
    CHECK(scores);
    const auto values = scores->values<float>();
    CHECK(values.size() == 2 && values[0] == 1.5f && values[1] == -2.0f);
    bool mismatch_thrown = false;
    try {
        scores->values<double>();
    } catch (const std::runtime_error&) {
        mismatch_thrown = true;
    }
    CHECK(mismatch_thrown);

    const brisk::TensorInfo* f32 = file.tensor("w.f32");
    CHECK(f32 && f32->type == TensorType::F32 && f32->n_elements() == 6);
    CHECK(f32->dims.size() == 2 && f32->dims[0] == 3 && f32->dims[1] == 2);
    CHECK(f32->data.size() == 24);
    float last = 0;
    std::memcpy(&last, f32->data.data() + 20, sizeof(last));
    CHECK(last == 5.0f);

    const brisk::TensorInfo* q4 = file.tensor("w.q4_0");
    CHECK(q4 && q4->data.size() == 36);  // 64 weights = 2 blocks of 18 bytes
    CHECK(q4->data.front() == std::byte{0} && q4->data.back() == std::byte{35});
    CHECK(file.tensor("missing") == nullptr);
}

// Every strict prefix of a valid file is truncated somewhere, so parsing it
// must throw rather than read out of bounds (run under ASan to verify).
void test_rejects_every_truncation() {
    const std::vector<std::byte> bytes = sample_file();
    for (std::size_t len = 0; len < bytes.size(); ++len) {
        CHECK(throws({bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(len)}));
    }
}

void test_rejects_malformed_input() {
    {
        std::vector<std::byte> bytes = sample_file();
        bytes[0] = std::byte{'X'};
        CHECK(throws(bytes));  // bad magic
    }
    {
        Builder b = header(0, 0);
        b.bytes()[4] = std::byte{9};
        CHECK(throws(b.bytes()));  // unsupported version
    }
    CHECK(throws(header(0, ~0ull).bytes()));  // metadata count far beyond file size
    CHECK(throws(header(~0ull, 0).bytes()));  // tensor count far beyond file size
    {
        Builder b = header(0, 1);
        b.str("k").put<std::uint32_t>(99);
        CHECK(throws(b.bytes()));  // unknown value type
    }
    {
        Builder b = header(0, 1);
        b.str("k").type(GgufType::String).put<std::uint64_t>(~0ull);
        CHECK(throws(b.bytes()));  // string length beyond file size
    }
    {
        Builder b = header(0, 1);
        b.str("k").type(GgufType::Array).type(GgufType::U64).put<std::uint64_t>(~0ull);
        CHECK(throws(b.bytes()));  // array byte size overflows
    }
    {
        Builder b = header(0, 2);
        b.str("k").type(GgufType::U8).put<std::uint8_t>(1);
        b.str("k").type(GgufType::U8).put<std::uint8_t>(2);
        CHECK(throws(b.bytes()));  // duplicate key
    }
    {
        Builder b = header(1, 0);
        b.str("t").put<std::uint32_t>(2).put<std::uint64_t>(1ull << 40).put<std::uint64_t>(1ull << 40);
        b.put(static_cast<std::uint32_t>(TensorType::F32)).put<std::uint64_t>(0);
        CHECK(throws(b.bytes()));  // element count overflows
    }
    {
        Builder b = header(1, 0);
        b.str("t").put<std::uint32_t>(1).put<std::uint64_t>(4);
        b.put(static_cast<std::uint32_t>(TensorType::F32)).put<std::uint64_t>(1ull << 60);
        b.pad_to(32).put(0.0f).put(0.0f).put(0.0f).put(0.0f);
        CHECK(throws(b.bytes()));  // tensor offset outside the file
    }
    {
        Builder b = header(1, 0);
        b.str("t").put<std::uint32_t>(1).put<std::uint64_t>(33);
        b.put(static_cast<std::uint32_t>(TensorType::Q4_0)).put<std::uint64_t>(0);
        b.pad_to(32);
        for (int i = 0; i < 64; ++i) b.put<std::uint8_t>(0);
        CHECK(throws(b.bytes()));  // 33 weights is not a whole Q4_0 block
    }
}

void test_keeps_unknown_tensor_types() {
    Builder b = header(1, 0);
    b.str("t").put<std::uint32_t>(1).put<std::uint64_t>(8);
    b.put<std::uint32_t>(999).put<std::uint64_t>(0);
    b.pad_to(32);
    const GgufFile file = GgufFile::parse(b.bytes());
    const brisk::TensorInfo* t = file.tensor("t");
    CHECK(t && t->data.empty() && to_string(t->type) == "unknown");
}

}  // namespace

int main() {
    test_parses_valid_file();
    test_rejects_every_truncation();
    test_rejects_malformed_input();
    test_keeps_unknown_tensor_types();
    std::printf("gguf_test: ok\n");
    return 0;
}
