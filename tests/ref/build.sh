#!/usr/bin/env bash
# Builds build/llama_ref against the llama.cpp checkout in baselines/llama.cpp.
# Run from the repo root after building llama.cpp there.
set -euo pipefail

LLAMA_DIR=${LLAMA_DIR:-baselines/llama.cpp}
lib_dir=$(cd "$LLAMA_DIR/build/bin" && pwd)

mkdir -p build
c++ -O2 -std=c++17 tests/ref/llama_ref.cpp \
    -I "$LLAMA_DIR/include" -I "$LLAMA_DIR/ggml/include" \
    -L "$lib_dir" -lllama -Wl,-rpath,"$lib_dir" \
    -o build/llama_ref
echo "built build/llama_ref"
