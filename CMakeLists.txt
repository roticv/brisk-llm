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

add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wconversion)
if(BRISK_SANITIZE)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined)
    add_link_options(-fsanitize=address,undefined)
endif()

find_package(Threads REQUIRED)

add_library(brisk
    src/gguf.cpp
    src/mapped_file.cpp
    src/model.cpp
    src/sampler.cpp
    src/tokenizer.cpp
    src/unicode.cpp
)
target_include_directories(brisk PUBLIC src)
target_link_libraries(brisk PUBLIC Threads::Threads)

add_executable(brisk-cli src/main.cpp)
set_target_properties(brisk-cli PROPERTIES OUTPUT_NAME brisk)
target_link_libraries(brisk-cli PRIVATE brisk)

add_executable(membw bench/membw.cpp)
target_link_libraries(membw PRIVATE Threads::Threads)

enable_testing()
add_executable(gguf_test tests/gguf_test.cpp)
target_link_libraries(gguf_test PRIVATE brisk)
add_test(NAME gguf_test COMMAND gguf_test)
