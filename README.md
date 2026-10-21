# brisk-llm

A small LLM inference engine in C++20, built for two machines: a MacBook Air M4 (CPU and Metal GPU) and a Raspberry Pi 4. It runs Qwen3, Qwen2.5, Llama 3.x, Gemma 3, Gemma 4 and BitNet b1.58 models from GGUF files and aims to match or beat llama.cpp on them. `ROADMAP.md` has the plan and where it stands; `bench/RESULTS.md` has the measurements.

## What it does

- Reads GGUF files directly (bounds-checked parser, fuzzed), using quantised weights in place from the memory map.
- Weight formats: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q8_0, Q4_K, Q6_K and the ternary TQ2_0; Q4_0, Q5_0 and TQ2_0 are repacked at load into a four-row interleaved layout on CPUs with `i8mm`.
- NEON kernels in three variants (baseline for the Cortex-A72, `dotprod`, `i8mm`/`smmla`), chosen at startup; batched prompt processing; a thread pool that hands out work dynamically so efficiency cores help.
- Metal backend (`--gpu`) for Q4_0, Q4_1, Q5_0, Q8_0, Q4_K, Q6_K and BitNet (TQ2_0) weights on Apple silicon: weights used in place from the memory map, K/V cache on the GPU, simdgroup-matrix kernels for prompts, one command buffer per batch; shaders compiled at startup, so no Metal toolchain is needed.
- Models: Qwen3 (0.6B, 1.7B), Qwen2.5 (1.5B) and the DeepSeek-R1-Distill-Qwen 1.5B, Llama 3.2 (1B, 3B), Mistral 7B v0.3, SmolLM2 1.7B, Gemma 3 (1B, stock Q4_K_M and pure Q4_0 files), Gemma 4 (E2B, E4B, 12B), Phi-4-mini, LFM2 (1.2B) and BitNet b1.58 2B4T — each checked against llama.cpp (tokenizer identical on the test corpus, perplexity, chat template) and pinned. Tokenizers: the Qwen, Llama 3, GPT-2/SmolLM and GPT-4o byte-level BPEs, the Llama/Mistral and Gemma 3 SentencePiece ones, Gemma 4's character BPE. `--chat` and the server apply each file's turn format.
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
build-release/brisk serve models/gemma-4-E2B-it-Q4_0.gguf --gpu --port 8080
```

`brisk serve` is an OpenAI-compatible API over one model: `POST /v1/chat/completions` (streamed or not), `/v1/completions`, `/v1/models`, `/health`. Point a chat front-end (Open WebUI, LibreChat, an editor plugin) or the OpenAI SDKs at `http://127.0.0.1:8080/v1` with any API key. Each family's chat template is applied (checked against llama.cpp's renderer in `tests/compare_chat.py`); `enable_thinking: false` (top level or in `chat_template_kwargs`) turns Qwen3's reasoning off; `temperature`, `top_p`, `top_k`, `seed`, `stop` and `max_tokens` are honoured. The session is kept between requests, so a conversation's next turn evaluates only the new tokens (`-v` logs how many were reused); one request runs at a time. `--ctx` sets the context the session holds (default: the model's, capped at 8192), `--host 0.0.0.0` exposes it on the network, and the Pi serves at its generation speed, 5-10 tokens/s on the 1B-2B models.

`--gpu` runs the model on the Metal GPU (Apple silicon): Q4_0, Q4_1, Q5_0, Q8_0, Q4_K, Q5_K and Q6_K matrices, so the common Q4_0 and Q4_K_M downloads run, and the BitNet TQ2_0 file. The weights are used in place from the memory-mapped file, the K/V cache lives on the GPU, and each batch of tokens is one command buffer. The shaders are embedded as source and compiled when the program starts, so no Metal toolchain is needed. `--warmup` evaluates the prompt once untimed first, as llama-bench does, for comparable numbers.

`brisk generate` prints a timing line to stderr; `BRISK_EVAL_CHUNK=1` makes `perplexity` score one token at a time (the generation kernels rather than the batched ones); `BRISK_PROFILE=1` adds a per-phase breakdown on the CPU, `BRISK_GPU_PROFILE=1` per kernel class on the GPU (each class timed in its own command buffer, so small ones read high), `BRISK_GPU_TIME=1` the GPU time of each batch, and for debugging `BRISK_GPU_LAYERS=<n>` evaluates only the first n layers while `BRISK_GPU_DEBUG=1` prints each intermediate buffer's magnitude and non-finite count and the last token's residual; `BRISK_CPU_DEBUG=1` prints the same residual after each layer on the CPU, so the two backends can be compared layer by layer. Model files go in `models/` (ignored by git); the pure Q4_0 files used in the benchmarks are made with `llama-quantize --pure`.

## Tests

```
ctest --test-dir build --output-on-failure         # unit tests (sanitizer build): parser, kernels, pool, tokenizer, drafter, Metal matmul
python3 tests/check_perplexity.py [--gpu]         # perplexity against pinned values, for the model files present
python3 tests/compare_tokenizer.py <model.gguf>   # token ids against llama.cpp
python3 tests/compare_perplexity.py <model.gguf>  # perplexity against llama.cpp
python3 tests/compare_chat.py                     # chat templates against llama.cpp, every model present
python3 tests/server_test.py                      # the API end to end on Qwen3-0.6B
```

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs the unit tests under the sanitizers on macOS and Linux ARM runners, replays the fuzz corpus, checks the pinned perplexity of a downloaded Qwen3-0.6B file and runs the server test on it. The Metal matmul test (every weight format, gemv and tile, against a float reference) and `check_perplexity.py --gpu` run on the macOS runner too. The GPU is checked against the CPU pins with a 2% tolerance: it keeps activations in float where the CPU quantises them to 8 bits, which is invisible on most models (0.1-0.3%) but worth 1.3% on Qwen2.5-1.5B, whose activation outliers the 8-bit path handles worse (there the GPU is also 1.1% below llama.cpp). After an intentional change to the numerics, re-pin with `python3 tests/check_perplexity.py --update` once `compare_perplexity.py` confirms agreement with llama.cpp. `BRISK_KERNELS=generic|neon|dotprod|i8mm` caps the kernel set in use, to exercise the other paths on a machine that has them all; the paths agree within about 0.5% in perplexity.

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

The CPU work (roadmap Phases 0-4), the Metal backend (Phase 5) and the model families (Phase 6) are done on both machines; the roadmap's Phase 7 lists the next families in three tiers (files to verify, Phi-4-mini/LFM2/SmolLM3, and the hybrid-attention and MoE models).

Against llama.cpp on the same files (details in `bench/RESULTS.md`):

- **CPU:** generation at 4 threads is within a few percent either way, because both engines sit at 80-90% of the memory-bandwidth ceiling. brisk is ahead on the Pi (generation 12%, prompt processing 1.9x on Qwen3-1.7B), on the M4 with all cores (40%), on Q4_K_M files, and on BitNet (2-3x); behind on M4 prompt processing of Q4_0 files at 4 threads (6%).
- **GPU:** level with llama.cpp's Metal backend on Qwen3-1.7B generation (89.6 vs 88.4 tokens/s), 11% ahead on BitNet generation (93 vs 84), and 3-9% behind on prompt processing (940 vs 1034 on the 1.7B), which is 2.9x the CPU path. The batched matmul runs at 85-90% of the GPU's measured matrix peak, so the rest of that gap is outside the kernel.
