#pragma once

namespace brisk {

struct CpuFeatures {
    bool dotprod = false;  // ARMv8.2 signed/unsigned 8-bit dot product (sdot/udot)
    bool i8mm = false;     // ARMv8.6 8-bit matrix multiply (smmla/usmmla)
};

// Detected once at first use.
const CpuFeatures& cpu_features();

}  // namespace brisk
