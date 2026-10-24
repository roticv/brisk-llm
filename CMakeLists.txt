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
    src/engine.cpp
    src/gguf.cpp
    src/kernels.cpp
    src/kernels_f32.cpp
    src/mapped_file.cpp
    src/model.cpp
    src/quant.cpp
    src/sampler.cpp
    src/chat.cpp
    src/memory.cpp
    src/speculative.cpp
    src/thread_pool.cpp
    src/tokenizer.cpp
    src/unicode.cpp
)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    # Each instruction-set variant lives in its own file built with its own
    # flags; the right one is picked at runtime (src/kernels.cpp).
    target_sources(brisk PRIVATE src/kernels_neon.cpp src/kernels_dotprod.cpp src/kernels_i8mm.cpp)
    set_source_files_properties(src/kernels_dotprod.cpp PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+dotprod")
    set_source_files_properties(src/kernels_i8mm.cpp PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+dotprod+i8mm")
endif()
target_include_directories(brisk PUBLIC src)
target_link_libraries(brisk PUBLIC Threads::Threads)

if(APPLE)
    option(BRISK_METAL "Build the Metal GPU backend" ON)
endif()
if(BRISK_METAL)
    enable_language(OBJCXX)
    set(CMAKE_OBJCXX_STANDARD 20)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
    # The shader source is embedded as a string and compiled by Metal at runtime.
    add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/generated/metal_shaders.inc
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/generated
        COMMAND ${CMAKE_COMMAND} -DIN=${CMAKE_SOURCE_DIR}/src/metal/kernels.metal
                -DOUT=${CMAKE_BINARY_DIR}/generated/metal_shaders.inc -P ${CMAKE_SOURCE_DIR}/cmake/embed_text.cmake
        DEPENDS src/metal/kernels.metal cmake/embed_text.cmake
        COMMENT "Embedding Metal shaders")
    target_sources(brisk PRIVATE src/metal/metal_session.mm ${CMAKE_BINARY_DIR}/generated/metal_shaders.inc)
    set_source_files_properties(src/metal/metal_session.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    target_include_directories(brisk PRIVATE ${CMAKE_BINARY_DIR}/generated)
    target_link_libraries(brisk PUBLIC "-framework Metal" "-framework Foundation")
    target_compile_definitions(brisk PUBLIC BRISK_METAL=1)
endif()

# The HTTP server: cpp-httplib and nlohmann/json, vendored in third_party/ (MIT).
add_library(brisk_server STATIC src/server.cpp third_party/cpp-httplib/httplib.cpp)
target_include_directories(brisk_server PUBLIC third_party/cpp-httplib third_party/nlohmann)
set_source_files_properties(third_party/cpp-httplib/httplib.cpp PROPERTIES COMPILE_OPTIONS "-w")  # vendored: its warnings are not ours
target_link_libraries(brisk_server PUBLIC brisk Threads::Threads)

add_executable(brisk-cli src/main.cpp)
set_target_properties(brisk-cli PROPERTIES OUTPUT_NAME brisk)
target_link_libraries(brisk-cli PRIVATE brisk_server)

add_executable(kernel_bench bench/kernel_bench.cpp)
target_link_libraries(kernel_bench PRIVATE brisk)

add_executable(attention_bench bench/attention_bench.cpp)
target_link_libraries(attention_bench PRIVATE brisk)

add_executable(membw bench/membw.cpp)
target_link_libraries(membw PRIVATE Threads::Threads)

enable_testing()
add_executable(gguf_test tests/gguf_test.cpp)
target_link_libraries(gguf_test PRIVATE brisk)
add_test(NAME gguf_test COMMAND gguf_test)
add_executable(kernels_test tests/kernels_test.cpp)
target_link_libraries(kernels_test PRIVATE brisk)
add_test(NAME kernels_test COMMAND kernels_test)
if(BRISK_METAL)
    add_executable(metal_test tests/metal_test.cpp)
    target_link_libraries(metal_test PRIVATE brisk)
    add_test(NAME metal_test COMMAND metal_test)
    add_executable(metal_bench bench/metal_bench.cpp)
    target_link_libraries(metal_bench PRIVATE brisk)
endif()
add_executable(thread_pool_test tests/thread_pool_test.cpp)
target_link_libraries(thread_pool_test PRIVATE brisk)
add_test(NAME thread_pool_test COMMAND thread_pool_test)
add_executable(tokenizer_test tests/tokenizer_test.cpp)
target_link_libraries(tokenizer_test PRIVATE brisk)
add_test(NAME tokenizer_test COMMAND tokenizer_test)
add_executable(speculative_test tests/speculative_test.cpp)
target_link_libraries(speculative_test PRIVATE brisk)
add_test(NAME speculative_test COMMAND speculative_test)

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
