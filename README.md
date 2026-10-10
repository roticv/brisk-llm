# brisk-llm

A small CPU inference engine for LLMs in C++20, built for two machines: a MacBook Air M4 and a Raspberry Pi 4. It runs Qwen3 and BitNet b1.58 models from GGUF files and aims to match or beat llama.cpp on them. `ROADMAP.md` has the plan and where it stands; `bench/RESULTS.md` has the measurements.

## What it does

- Reads GGUF files directly (bounds-checked parser, fuzzed), using quantised weights in place from the memory map.
- Weight formats: F32, F16, BF16, Q4_0, Q4_1, Q4_K, Q6_K and the ternary TQ2_0; Q4_0 and TQ2_0 are repacked at load into a four-row interleaved layout on CPUs with `i8mm`.
- NEON kernels in three variants (baseline for the Cortex-A72, `dotprod`, `i8mm`/`smmla`), chosen at startup; batched prompt processing; a thread pool that hands out work dynamically so efficiency cores help.
- Models: Qwen3 (0.6B, 1.7B tested) and BitNet b1.58 2B4T, with the Qwen and Llama 3 tokenizers.
- Greedy and temperature/top-k/top-p sampling; speculative decoding with lookup (n-gram) or model drafting, exact with respect to greedy decoding.

## Build

Needs CMake 3.20+, Ninja and a C++20 compiler (Apple clang 17 and GCC 14 are used).

```
cmake -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-release
```

A sanitizer build for development: `cmake -B build -DBRISK_SANITIZE=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo`.

## Run

```
build-release/brisk info models/Qwen3-1.7B-pure-Q4_0.gguf
build-release/brisk generate models/Qwen3-1.7B-pure-Q4_0.gguf --chat --no-think -p "Name three primary colours."
build-release/brisk generate models/Qwen3-1.7B-pure-Q4_0.gguf -f prompt.txt -n 200 -t 4 --draft lookup
build-release/brisk perplexity models/Qwen3-1.7B-pure-Q4_0.gguf -f text.txt
```

`brisk generate` prints a timing line to stderr; `BRISK_PROFILE=1` adds a per-phase breakdown. Model files go in `models/` (ignored by git); the pure Q4_0 files used in the benchmarks are made with `llama-quantize --pure`.

## Tests

```
build/gguf_test && build/kernels_test            # unit tests (sanitizer build)
python3 tests/compare_tokenizer.py <model.gguf>   # token ids against llama.cpp
python3 tests/compare_perplexity.py <model.gguf>  # perplexity against llama.cpp
```

The comparison scripts need `build/llama_ref`, built by `tests/ref/build.sh` against a llama.cpp checkout in `baselines/llama.cpp`. Perplexity is the authoritative check for quantised models: 8-bit activation rounding makes per-logit comparison chaotic (llama.cpp's own kernel variants disagree by up to 2 in a logit), while perplexity agrees within about 1% and float32 matches to four decimals.

## Benchmarks

```
bench/run_brisk_bench.sh     # brisk: ~500-token prompt, 128 generated tokens, medians of 5
bench/run_llama_bench.sh     # llama-bench with the same settings
build-release/membw          # memory read bandwidth, the ceiling for generation
build-release/kernel_bench   # the Q4_0 kernels in isolation
```

Benchmarks on the Air are only meaningful with nothing else running; the fanless chassis throttles under sustained compute, so medians of several runs are reported.

## Status

Phases 0-2 of the roadmap are done; Phase 3 stopped after BitNet and speculative decoding, with the findings recorded in the roadmap. In short: generation at 4 threads sits within a few percent of llama.cpp either way, because both engines are at 80-90% of the memory-bandwidth ceiling; brisk is ahead on the Pi, on the M4 with all cores (where it also passes llama.cpp's Metal path), on Q4_K_M files, and on BitNet (2-3x), and behind on M4 prompt processing of Q4_0 files at 4 threads.
