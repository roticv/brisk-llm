#pragma once

#include <cstddef>
#include <span>
#include <string>

namespace brisk {

// Read-only memory mapping of a whole file. Weights are used straight from the
// mapping, so the OS can page them in on demand and drop them under pressure.
class MappedFile {
public:
    explicit MappedFile(const std::string& path);
    ~MappedFile();

    MappedFile(MappedFile&& other) noexcept;
    MappedFile& operator=(MappedFile&& other) noexcept;
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    std::span<const std::byte> bytes() const { return {data_, size_}; }

    // Pages the whole file in now (reading it once, after telling the OS to
    // read ahead), so the first use does not pay for it; `lock` also pins the
    // pages (mlock), which fails quietly where the limit forbids it. Returns
    // the bytes touched.
    std::size_t prefetch(bool lock = false) const;

private:
    void reset() noexcept;

    const std::byte* data_ = nullptr;
    std::size_t size_ = 0;
};

}  // namespace brisk
