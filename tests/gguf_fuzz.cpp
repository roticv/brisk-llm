// libFuzzer target for the GGUF parser: any input must either parse or throw
// std::runtime_error, never crash or read out of bounds.
//
// Build (needs a clang with libFuzzer, e.g. Homebrew LLVM on macOS):
//   CXX=/opt/homebrew/opt/llvm/bin/clang++ cmake -B build-fuzz -DBRISK_FUZZ=ON -DBRISK_SANITIZE=ON
//   cmake --build build-fuzz --target gguf_fuzz
// Run:   build-fuzz/gguf_fuzz -max_len=4096 -max_total_time=60 tests/fuzz_corpus

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

#include "gguf.h"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    const std::span<const std::byte> bytes(reinterpret_cast<const std::byte*>(data), size);
    try {
        const brisk::GgufFile file = brisk::GgufFile::parse(bytes);
        // Exercise the accessors too: they must be safe on anything that parsed.
        for (const auto& [key, value] : file.metadata()) {
            if (const auto* array = std::get_if<brisk::GgufArray>(&value)) {
                if (array->elem_type == brisk::GgufType::String) array->strings();
                else if (array->elem_type == brisk::GgufType::F32) array->values<float>();
            }
        }
        for (const brisk::TensorInfo& t : file.tensors()) t.n_elements();
    } catch (const std::runtime_error&) {
    }
    return 0;
}
