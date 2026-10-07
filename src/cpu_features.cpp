#include "cpu_features.h"

#include <cstddef>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sys/auxv.h>
#endif

namespace brisk {

namespace {

#if defined(__APPLE__)
bool sysctl_flag(const char* name) {
    int value = 0;
    std::size_t size = sizeof(value);
    return sysctlbyname(name, &value, &size, nullptr, 0) == 0 && value != 0;
}
#endif

CpuFeatures detect() {
    CpuFeatures f;
#if !defined(__aarch64__)
    return f;
#elif defined(__APPLE__)
    f.dotprod = sysctl_flag("hw.optional.arm.FEAT_DotProd");
    f.i8mm = sysctl_flag("hw.optional.arm.FEAT_I8MM");
#elif defined(__linux__)
    // Bit positions from the Linux kernel's arm64 hwcap definitions.
    constexpr unsigned long kHwcapAsimdDp = 1ul << 20;
    constexpr unsigned long kHwcap2I8mm = 1ul << 13;
    f.dotprod = (getauxval(AT_HWCAP) & kHwcapAsimdDp) != 0;
    f.i8mm = (getauxval(AT_HWCAP2) & kHwcap2I8mm) != 0;
#endif
    return f;
}

}  // namespace

const CpuFeatures& cpu_features() {
    static const CpuFeatures features = detect();
    return features;
}

}  // namespace brisk
