#include "memory.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>

#include "kernels.h"
#include "quant.h"

#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

namespace brisk {

namespace {

std::string gb(std::size_t bytes) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.2f GB", static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
    return buf;
}

// Whether the CPU path makes a grouped copy of a matrix of this type (Model::load_matrix).
bool repacks(TensorType type) {
    switch (type) {
        case TensorType::Q4_0:
        case TensorType::Q5_0:
        case TensorType::Q4_K:
        case TensorType::Q6_K:
        case TensorType::TQ2_0: return true;
        default: return false;
    }
}

}  // namespace

std::string MemoryEstimate::describe() const {
    return gb(total()) + " (" + gb(weights) + " weights, " + gb(repacked) + " repacked copies, " + gb(cache) + " cache, " +
           gb(scratch) + " scratch)";
}

MemoryEstimate estimate_memory(const GgufFile& file, const ModelConfig& c, bool repack, std::size_t max_context) {
    MemoryEstimate e;
    const bool grouped = repack && kernels::prefers_q4_0x4();
    for (const TensorInfo& t : file.tensors()) {
        e.weights += t.data.size();
        // Matrices (2-D, a multiple of 4 rows) of the repacked formats get a same-size copy;
        // the token embedding is read by rows and the per-layer table too, so neither does.
        if (grouped && t.dims.size() == 2 && t.dims[1] % 4 == 0 && repacks(t.type) && t.name != "token_embd.weight" &&
            t.name != "per_layer_token_embd.weight") {
            e.repacked += t.data.size();
        }
        if (t.type == TensorType::F16 || t.type == TensorType::BF16) e.repacked += t.data.size();  // widened to float
    }
    const std::size_t context = std::min<std::size_t>(max_context == 0 ? c.context_length : max_context, c.context_length);
    for (std::size_t l = 0; l < c.layer_count; ++l) {
        // The caches hold halves.
        if (c.layer_is_conv(l)) {
            e.cache += context * c.embedding_dim * 2;
        } else if (c.kv_layer_of(l) == l) {
            e.cache += 2 * std::size_t{c.kv_head_count_of(l)} * c.head_dim_of(l) * context * 2;
        }
    }
    // Scratch, following each backend's allocations (Session and MetalSession).
    const std::size_t qdim = std::size_t{c.head_count} * c.head_dim;
    const std::size_t kvdim = std::size_t{c.kv_head_count} * c.head_dim;
    if (repack) {
        // CPU Session: a batch of 256 tokens; the scores are allocated per batch over the context.
        constexpr std::size_t batch = 256;
        e.scratch += batch * (4 * c.embedding_dim + 2 * qdim + kvdim + 2 * c.feed_forward_dim) * sizeof(float);
        e.scratch += batch * c.head_count * context * sizeof(float);  // scores
        e.scratch += 64 * c.vocab_size * sizeof(float);              // logits for scored tokens
        if (c.ple_dim != 0) e.scratch += batch * std::size_t{c.ple_dim} * c.layer_count * sizeof(float) * 2;
        if (c.conv_taps != 0) e.scratch += batch * 3 * c.embedding_dim * sizeof(float);
    } else {
        // MetalSession: a batch of 512 tokens with float and half copies of the activations,
        // the batched attention's scores (float) and probabilities (half) for the batch over
        // the context, the small kernel's partials, and the logits.
        constexpr std::size_t batch = 512;
        e.scratch += batch * (4 * c.embedding_dim + 2 * qdim + 2 * kvdim + c.feed_forward_dim) * sizeof(float);
        e.scratch += batch * (c.embedding_dim + qdim + c.feed_forward_dim) * 2;  // half copies
        e.scratch += batch * c.head_count * context * 6;                         // scores + probabilities
        e.scratch += 8 * c.head_count * ((context + 127) / 128) * (c.head_dim + 2) * sizeof(float);  // partials
        e.scratch += 64 * c.vocab_size * sizeof(float);
        if (c.ple_dim != 0) e.scratch += batch * std::size_t{c.ple_dim} * c.layer_count * sizeof(float) * 3;
        if (c.conv_taps != 0) e.scratch += batch * 3 * c.embedding_dim * sizeof(float);
    }
    return e;
}

std::size_t available_memory() {
#if defined(__APPLE__)
    vm_statistics64_data_t vm;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm), &count) != KERN_SUCCESS) return 0;
    vm_size_t page = 0;
    if (host_page_size(mach_host_self(), &page) != KERN_SUCCESS) return 0;
    // Free, inactive (reclaimable) and the file-backed pages the kernel can drop.
    return static_cast<std::size_t>(vm.free_count + vm.inactive_count + vm.external_page_count) * page;
#else
    std::ifstream in("/proc/meminfo");
    std::string key;
    std::size_t value = 0;
    std::string unit;
    while (in >> key >> value >> unit) {
        if (key == "MemAvailable:") return value * 1024;
    }
    return 0;
#endif
}

void check_memory(const MemoryEstimate& e, std::size_t max_context, const std::string& what) {
    if (std::getenv("BRISK_NO_MEMORY_CHECK") != nullptr) return;
    const std::size_t available = available_memory();
    if (available == 0) return;
    constexpr std::size_t margin = std::size_t{256} << 20;  // left to the OS and the page cache
    if (e.total() + margin <= available) return;
    const std::size_t shortfall = e.total() + margin - available;
    std::string hint = "a smaller file or quantisation";
    // A shorter context frees most of the cache (and some scratch); say so when that alone would do.
    if (shortfall <= e.cache * 7 / 8 && max_context > 512) {
        hint = "a shorter context (--ctx; this estimate is for " + std::to_string(max_context) + " tokens and the cache is " +
               gb(e.cache) + "), or " + hint;
    }
    throw std::runtime_error(what + " needs about " + e.describe() + " but " + gb(available) +
                             " is available; running anyway would swap. Try " + hint +
                             ", or set BRISK_NO_MEMORY_CHECK=1 to proceed regardless.");
}

}  // namespace brisk
