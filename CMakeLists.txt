cmake_minimum_required(VERSION 3.20)
project(brisk-llm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(BRISK_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(BRISK_FUZZ "Build the libFuzzer targets (needs clang)" OFF)

add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wconversion)
if(BRISK_SANITIZE)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined)
    add_link_options(-fsanitize=address,undefined)
endif()

find_package(Threads REQUIRED)

add_library(brisk
    src/cpu_features.cpp
    src/gguf.cpp
    src/kernels.cpp
    src/mapped_file.cpp
    src/model.cpp
    src/quant.cpp
    src/sampler.cpp
    src/thread_pool.cpp
    src/tokenizer.cpp
    src/unicode.cpp
)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    # Each instruction-set variant lives in its own file built with its own
    # flags; the right one is picked at runtime (src/kernels.cpp).
    target_sources(brisk PRIVATE src/kernels_neon.cpp src/kernels_dotprod.cpp)
    set_source_files_properties(src/kernels_dotprod.cpp PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+dotprod")
endif()
target_include_directories(brisk PUBLIC src)
target_link_libraries(brisk PUBLIC Threads::Threads)

add_executable(brisk-cli src/main.cpp)
set_target_properties(brisk-cli PROPERTIES OUTPUT_NAME brisk)
target_link_libraries(brisk-cli PRIVATE brisk)

add_executable(kernel_bench bench/kernel_bench.cpp)
target_link_libraries(kernel_bench PRIVATE brisk)

add_executable(membw bench/membw.cpp)
target_link_libraries(membw PRIVATE Threads::Threads)

enable_testing()
add_executable(gguf_test tests/gguf_test.cpp)
target_link_libraries(gguf_test PRIVATE brisk)
add_test(NAME gguf_test COMMAND gguf_test)
add_executable(kernels_test tests/kernels_test.cpp)
target_link_libraries(kernels_test PRIVATE brisk)
add_test(NAME kernels_test COMMAND kernels_test)

if(BRISK_FUZZ)
    add_executable(gguf_fuzz tests/gguf_fuzz.cpp)
    target_link_libraries(gguf_fuzz PRIVATE brisk)
    target_compile_options(gguf_fuzz PRIVATE -fsanitize=fuzzer)
    target_link_options(gguf_fuzz PRIVATE -fsanitize=fuzzer)
    # A non-Apple clang (Homebrew LLVM) compiles against its own libc++ headers,
    # so it must link its own libc++ too, not the system one.
    get_filename_component(_llvm_root "${CMAKE_CXX_COMPILER}" DIRECTORY)
    get_filename_component(_llvm_root "${_llvm_root}" DIRECTORY)
    if(APPLE AND EXISTS "${_llvm_root}/lib/c++")
        target_link_options(gguf_fuzz PRIVATE "-L${_llvm_root}/lib/c++" "-Wl,-rpath,${_llvm_root}/lib/c++")
    endif()
endif()
