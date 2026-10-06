#include "mapped_file.h"

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace brisk {

namespace {

[[noreturn]] void fail(const std::string& what, const std::string& path) {
    throw std::runtime_error(what + " '" + path + "': " + std::strerror(errno));
}

}  // namespace

MappedFile::MappedFile(const std::string& path) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) fail("cannot open", path);

    struct stat st {};
    if (::fstat(fd, &st) != 0) {
        const int saved = errno;
        ::close(fd);
        errno = saved;
        fail("cannot stat", path);
    }
    if (st.st_size <= 0) {
        ::close(fd);
        throw std::runtime_error("file is empty: '" + path + "'");
    }

    const auto size = static_cast<std::size_t>(st.st_size);
    void* addr = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    const int saved = errno;
    ::close(fd);
    if (addr == MAP_FAILED) {
        errno = saved;
        fail("cannot mmap", path);
    }

    data_ = static_cast<const std::byte*>(addr);
    size_ = size;
}

MappedFile::~MappedFile() { reset(); }

MappedFile::MappedFile(MappedFile&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
    if (this != &other) {
        reset();
        data_ = std::exchange(other.data_, nullptr);
        size_ = std::exchange(other.size_, 0);
    }
    return *this;
}

void MappedFile::reset() noexcept {
    if (data_ != nullptr) {
        ::munmap(const_cast<std::byte*>(data_), size_);
        data_ = nullptr;
        size_ = 0;
    }
}

}  // namespace brisk
