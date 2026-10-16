# brisk-llm

A small LLM inference engine in C++20, built for two machines: a MacBook Air M4 (CPU and Metal GPU) and a Raspberry Pi 4. It runs Qwen3 and BitNet b1.58 models from GGUF files and aims to match or beat llama.cpp on them. `ROADMAP.md` has the plan and where it stands; `bench/RESULTS.md` has the measurements.

## What it does

- Reads GGUF files directly (bounds-checked parser, fuzzed), using quantised weights in place from the memory map.
- Weight formats: F32, F16, BF16, Q4_0, Q4_1, Q4_K, Q6_K and the ternary TQ2_0; Q4_0 and TQ2_0 are repacked at load into a four-row interleaved layout on CPUs with `i8mm`.
- NEON kernels in three variants (baseline for the Cortex-A72, `dotprod`, `i8mm`/`smmla`), chosen at startup; batched prompt processing; a thread pool that hands out work dynamically so efficiency cores help.
- Metal backend (`--gpu`) for Q4_0, Q4_1, Q4_K, Q6_K and BitNet (TQ2_0) weights on Apple silicon: weights used in place from the memory map, K/V cache on the GPU, simdgroup-matrix kernels for prompts, one command buffer per batch; shaders compiled at startup, so no Metal toolchain is needed.
- Models: Qwen3 (0.6B, 1.7B tested), Qwen2/2.5 (1.5B tested; also the DeepSeek-R1 distils of it), Llama 3.x (3.2-1B tested; Mistral- and SmolLM2-shaped files should load) and BitNet b1.58 2B4T, with the Qwen and Llama 3 tokenizers. `--chat` applies each family's turn format. Gemma is planned (roadmap Phase 6).
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
build-release/brisk generate models/Qwen3-1.7B-pure-Q4_0.gguf -f prompt.txt -n 200 --gpu
```

`--gpu` runs the model on the Metal GPU (Apple silicon): Q4_0, Q4_1, Q4_K and Q6_K matrices, so the common Q4_0 and Q4_K_M downloads run, and the BitNet TQ2_0 file. The weights are used in place from the memory-mapped file, the K/V cache lives on the GPU, and each batch of tokens is one command buffer. The shaders are embedded as source and compiled when the program starts, so no Metal toolchain is needed. `--warmup` evaluates the prompt once untimed first, as llama-bench does, for comparable numbers.

`brisk generate` prints a timing line to stderr; `BRISK_EVAL_CHUNK=1` makes `perplexity` score one token at a time (the generation kernels rather than the batched ones); `BRISK_PROFILE=1` adds a per-phase breakdown on the CPU, `BRISK_GPU_PROFILE=1` per kernel class on the GPU (each class timed in its own command buffer, so small ones read high) and `BRISK_GPU_TIME=1` the GPU time of each batch. Model files go in `models/` (ignored by git); the pure Q4_0 files used in the benchmarks are made with `llama-quantize --pure`.

## Tests

```
ctest --test-dir build --output-on-failure         # unit tests (sanitizer build): parser, kernels, pool, tokenizer, drafter, Metal matmul
python3 tests/check_perplexity.py [--gpu]         # perplexity against pinned values, for the model files present
python3 tests/compare_tokenizer.py <model.gguf>   # token ids against llama.cpp
python3 tests/compare_perplexity.py <model.gguf>  # perplexity against llama.cpp
```

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs the unit tests under the sanitizers on macOS and Linux ARM runners, replays the fuzz corpus, and checks the pinned perplexity of a downloaded Qwen3-0.6B file The Metal matmul test (every weight format, gemv and tile, against a float reference) and `check_perplexity.py --gpu` run on the macOS runner too. The GPU is checked against the CPU pins with a 2% tolerance: it keeps activations in float where the CPU quantises them to 8 bits, which is invisible on most models (0.1-0.3%) but worth 1.3% on Qwen2.5-1.5B, whose activation outliers the 8-bit path handles worse (there the GPU is also 1.1% below llama.cpp). After an intentional change to the numerics, re-pin with `python3 tests/check_perplexity.py --update` once `compare_perplexity.py` confirms agreement with llama.cpp. `BRISK_KERNELS=generic|neon|dotprod|i8mm` caps the kernel set in use, to exercise the other paths on a machine that has them all; the paths agree within about 0.5% in perplexity.

The comparison scripts need `build/llama_ref`, built by `tests/ref/build.sh` against a llama.cpp checkout in `baselines/llama.cpp`. Perplexity is the authoritative check for quantised models: 8-bit activation rounding makes per-logit comparison chaotic (llama.cpp's own kernel variants disagree by up to 2 in a logit), while perplexity agrees within about 1% and float32 matches to four decimals.

## Benchmarks

```
bench/run_brisk_bench.sh     # brisk: ~500-token prompt, 128 generated tokens, medians of 5
bench/run_llama_bench.sh     # llama-bench with the same settings
build-release/membw          # memory read bandwidth, the ceiling for generation
build-release/kernel_bench   # the Q4_0 kernels in isolation
build-release/metal_bench    # the Metal Q4_0 kernels: GFLOPS and weight bandwidth per shape
```

Benchmarks on the Air are only meaningful with nothing else running; the fanless chassis throttles under sustained compute, so medians of several runs are reported.

## Status

The CPU work (roadmap Phases 0-4) and the Metal backend (Phase 5) are done; Phase 6, more model families, is next.

Against llama.cpp on the same files (details in `bench/RESULTS.md`):

- **CPU:** generation at 4 threads is within a few percent either way, because both engines sit at 80-90% of the memory-bandwidth ceiling. brisk is ahead on the Pi (generation 12%, prompt processing 1.9x on Qwen3-1.7B), on the M4 with all cores (40%), on Q4_K_M files, and on BitNet (2-3x); behind on M4 prompt processing of Q4_0 files at 4 threads (6%).
- **GPU:** level with llama.cpp's Metal backend on Qwen3-1.7B generation (89.6 vs 88.4 tokens/s), 11% ahead on BitNet generation (93 vs 84), and 3-9% behind on prompt processing (940 vs 1034 on the 1.7B), which is 2.9x the CPU path. The batched matmul runs at 85-90% of the GPU's measured matrix peak, so the rest of that gap is outside the kernel.
