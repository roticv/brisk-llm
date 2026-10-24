#pragma once
// What a model needs in memory, before anything is allocated, against what
// the machine has: a file that does not fit swaps instead of failing, which
// on a Raspberry Pi turns into an unusable board rather than an error.
#include <cstddef>
#include <string>

#include "gguf.h"
#include "model.h"

namespace brisk {

struct MemoryEstimate {
    std::size_t weights = 0;   // the file's tensors (mapped; resident while in use)
    std::size_t repacked = 0;  // copies of the matrices in the CPU kernels' grouped layouts
    std::size_t cache = 0;     // K/V (and convolution) state for the context
    std::size_t scratch = 0;   // activation buffers, scores, logits
    std::size_t total() const { return weights + repacked + cache + scratch; }
    std::string describe() const;  // "3.4 GB: 2.8 weights, ..."
};

// The estimate for running `file` with a session of `max_context` tokens on
// the CPU (repack = true; the GPU takes the weights in place).
MemoryEstimate estimate_memory(const GgufFile& file, const ModelConfig& config, bool repack, std::size_t max_context);

// Memory the process could use now without swapping: free plus reclaimable
// page cache, as the OS reports it. 0 when unknown.
std::size_t available_memory();

// Throws std::runtime_error naming the shortfall and a way round it when the
// estimate (plus a margin for the OS) exceeds the available memory; `what`
// is the command, for the message. No-op when BRISK_NO_MEMORY_CHECK is set.
void check_memory(const MemoryEstimate& estimate, std::size_t max_context, const std::string& what);

}  // namespace brisk
