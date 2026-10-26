# brisk-llm

A small LLM inference engine in C++20, built for two machines: a MacBook Air M4 (CPU and Metal GPU) and a Raspberry Pi 4. It runs Qwen3, Qwen2.5, Llama 3.x, Gemma 3, Gemma 4 and BitNet b1.58 models from GGUF files and aims to match or beat llama.cpp on them. `ROADMAP.md` has the plan and where it stands; `bench/RESULTS.md` has the measurements.

## What it does

- Reads GGUF files directly (bounds-checked parser, fuzzed), using quantised weights in place from the memory map.
- Weight formats: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K and the ternary TQ2_0. At load, matrices are repacked into the layout the machine's kernels want: four-row interleaved on CPUs with `i8mm`, and for ternary weights on CPUs without it a 16-row lookup-table layout (bitnet.cpp's method: per-token tables of pair sums read with `tbl`, no multiplier).
- NEON kernels in three variants (baseline for the Cortex-A72, `dotprod`, `i8mm`/`smmla`), chosen at startup; batched prompt processing with banded, multi-token kernels for every format; a half-precision K/V cache; a thread pool that hands out work dynamically so efficiency cores help.
- Metal backend (`--gpu`) for all of the above formats on Apple silicon: weights used in place from the memory map, K/V cache on the GPU, simdgroup-matrix kernels for prompts, one command buffer per batch; shaders compiled at startup, so no Metal toolchain is needed.
- An OpenAI-compatible server (`brisk serve`), a memory preflight that refuses a model that would swap, and `brisk pull` for the verified model files.
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
build-release/brisk pull                      # lists the verified model files
build-release/brisk pull qwen3-1.7b lfm2-1.2b # downloads them into models/ (resumable; needs curl)
build-release/brisk info models/Qwen3-1.7B-Q4_0.gguf
build-release/brisk info --memory models/Qwen3-1.7B-Q4_0.gguf 4096   # what a 4096-token session needs, CPU and GPU
build-release/brisk generate models/Qwen3-1.7B-Q4_0.gguf --chat --no-think -p "Name three primary colours."
build-release/brisk generate models/Qwen3-1.7B-Q4_0.gguf -f prompt.txt -n 200 -t 4 --draft lookup
build-release/brisk generate models/Qwen3-1.7B-Q4_0.gguf -f prompt.txt -n 200 --gpu
build-release/brisk perplexity models/Qwen3-1.7B-Q4_0.gguf -f text.txt
build-release/brisk serve models/gemma-4-E2B-it-Q4_0.gguf --gpu --port 8080
```

### On a Raspberry Pi 4

```
sudo apt install cmake ninja-build g++ curl          # Debian 13 / Raspberry Pi OS 64-bit
cmake -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build-release
build-release/brisk pull lfm2-1.2b                   # 0.7 GB; or qwen3-0.6b, gemma-3-1b, llama-3.2-1b
build-release/brisk serve models/LFM2-1.2B-Q4_0.gguf --host 0.0.0.0 --ctx 2048
```

Then point a chat front-end on another machine at `http://raspberrypi.local:8080/v1`. The 1B-2B files answer at 4-10 tokens/s (`bench/RESULTS.md` has every file); a 4 GB Pi holds the 1B-2B files comfortably and Gemma 4 E2B with `--ctx 1024`. Before loading, `generate`, `perplexity` and `serve` estimate what the session needs against the memory available and refuse with the breakdown and a fix (`--ctx`, a smaller file) when it would not fit, since a model that does not fit swaps rather than fails and a Pi swapping to its SD card is unusable (`BRISK_NO_MEMORY_CHECK=1` skips the check). The server pages the weights in and runs a warm-up token before it listens, so the first request does not wait for the SD card.

### The server

`brisk serve` is an OpenAI-compatible API over one model: `POST /v1/chat/completions` (streamed or not), `/v1/completions`, `/v1/models`, `/health`. Point a chat front-end (Open WebUI, LibreChat, an editor plugin) or the OpenAI SDKs at `http://127.0.0.1:8080/v1` with any API key. Each file's chat template is applied (checked against llama.cpp's renderer in `tests/compare_chat.py`); `tools` are rendered for Qwen and Llama 3 files (the other templates have no tool format) and the model's calls come back as `tool_calls` with `finish_reason: "tool_calls"`, streamed as one delta at the end; `tool` messages and assistant `tool_calls` render for the next round; a `<think>` block (Qwen3, DeepSeek-R1) is returned as `reasoning_content` on the message and in the streamed deltas, with `content` holding only the answer, as llama-server and the DeepSeek API do (`"reasoning_format": "inline"` keeps it in `content`); `enable_thinking: false` (top level or in `chat_template_kwargs`) turns Qwen3's reasoning off; `temperature`, `top_p`, `top_k`, `seed`, `stop` and `max_tokens` are honoured. The session is kept between requests, so a conversation's next turn evaluates only the new tokens (`-v` logs how many were reused); one request runs at a time. `--ctx` sets the context the session holds (default: the model's, capped at 8192), `--host 0.0.0.0` exposes it on the network, `--no-prefetch` skips the page-in at start and `--mlock` pins the weights. SIGINT or SIGTERM stops it cleanly: a streamed answer in flight gets its finish chunk and `[DONE]`, then the process exits.

### The GPU

`--gpu` runs the model on the Metal GPU (Apple silicon): every weight format brisk reads, so the common Q4_0 and Q4_K_M downloads run, and the BitNet TQ2_0 file. The weights are used in place from the memory-mapped file, the K/V cache (half precision on both backends, as in llama.cpp) lives on the GPU, and each batch of tokens is one command buffer. The shaders are embedded as source and compiled when the program starts, so no Metal toolchain is needed. `--warmup` evaluates the prompt once untimed first, as llama-bench does, for comparable numbers.

### Diagnostics

`brisk generate` prints a timing line to stderr. `BRISK_EVAL_CHUNK=1` makes `perplexity` score one token at a time (the generation kernels rather than the batched ones); `BRISK_PROFILE=1` adds a per-phase breakdown on the CPU, `BRISK_GPU_PROFILE=1` per kernel class on the GPU (each class timed in its own command buffer, so small ones read high), `BRISK_GPU_TIME=1` the GPU time of each batch; `BRISK_GPU_LAYERS=<n>` evaluates only the first n layers while `BRISK_GPU_DEBUG=1` prints each intermediate buffer's magnitude and non-finite count and the last token's residual, and `BRISK_CPU_DEBUG=1` prints the same residual after each layer on the CPU, so the two backends can be compared layer by layer. `BRISK_KERNELS=generic|neon|dotprod|i8mm` caps the kernel set in use. Model files go in `models/` (ignored by git); `brisk pull` fetches the ones brisk is checked against (`src/catalog.cpp`: a published GGUF per family), and the pure Q4_0 files used in some benchmarks are made with `llama-quantize --pure`.

## Tests

```
ctest --test-dir build --output-on-failure         # unit tests (sanitizer build): parser, kernels, pool, tokenizer, drafter, Metal matmul
python3 tests/check_perplexity.py [--gpu]         # perplexity against pinned values, for the model files present
python3 tests/compare_tokenizer.py <model.gguf>   # token ids against llama.cpp
python3 tests/compare_perplexity.py <model.gguf>  # perplexity against llama.cpp
python3 tests/compare_chat.py                     # chat templates against llama.cpp, every model present
python3 tests/server_test.py                      # the API end to end on Qwen3-0.6B
```

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs the unit tests under the sanitizers on macOS and Linux ARM runners, replays the fuzz corpus, checks the pinned perplexity of a downloaded Qwen3-0.6B file and runs the server test on it. The Metal matmul test (every weight format, gemv and tile, against a float reference) and `check_perplexity.py --gpu` run on the macOS runner too. The GPU is checked against the CPU pins with a 2% tolerance: it keeps activations in float where the CPU quantises them to 8 bits, which is invisible on most models (0.1-0.3%) but worth 1.3% on Qwen2.5-1.5B, whose activation outliers the 8-bit path handles worse (there the GPU is also 1.1% below llama.cpp). After an intentional change to the numerics, re-pin with `python3 tests/check_perplexity.py --update` once `compare_perplexity.py` confirms agreement with llama.cpp. The kernel sets (`BRISK_KERNELS`) agree within about 0.5% in perplexity.

The comparison scripts need `build/llama_ref`, built by `tests/ref/build.sh` against a llama.cpp checkout in `baselines/llama.cpp`. Perplexity is the authoritative check for quantised models: 8-bit activation rounding makes per-logit comparison chaotic (llama.cpp's own kernel variants disagree by up to 2 in a logit), while perplexity agrees within about 1% and float32 matches to four decimals.

## Benchmarks

```
bench/run_brisk_bench.sh     # brisk: ~500-token prompt, 128 generated tokens, medians of 5
bench/run_llama_bench.sh     # llama-bench with the same settings
build-release/membw          # memory read bandwidth, the ceiling for generation
build-release/kernel_bench   # the Q4_0 kernels in isolation
build-release/kquant_bench   # the K-quant batched kernels per core
build-release/ternary_lut_bench  # the ternary kernels: vmull chain against the lookup table
build-release/metal_bench    # the Metal Q4_0 kernels: GFLOPS and weight bandwidth per shape
bench/run_mac_sweep.sh       # every file, both backends, against llama-bench, as one table
```

Benchmarks on the Air are only meaningful with nothing else running; the fanless chassis throttles under sustained compute, so medians of several runs are reported.

## Status

Phases 0-6 of the roadmap are done on both machines (the CPU engine, the Metal backend, the model families), Phase 7's first two tiers (Phi-4-mini, LFM2 and the verified downloads), Phase 8 (the server) and most of Phase 9 (the Pi: memory preflight, half-precision cache, prefetch, `pull`, the lookup-table ternary kernel). Open: SmolLM3 and the hybrid-attention families, aarch64 release binaries, and the tuning items the roadmap lists.

Against llama.cpp on the same files (`bench/RESULTS.md`, Summary):

- **Raspberry Pi 4:** ahead on every file. Prompt processing 1.2-1.9x (Q4_0 files 1.7-1.9x, K-quant files 1.3-1.5x, BitNet 1.2x), generation 2-9% ahead with both engines at the memory-bandwidth ceiling.
- **M4 CPU (4 threads):** level on generation everywhere (bandwidth-bound); prompts 10-30% ahead on the K-quant files, level on Llama and Phi, 6-12% behind on the Q4_0 and LFM2 files. With all ten cores brisk's generation is 40% ahead because the pool uses the efficiency cores.
- **M4 GPU:** generation within a few percent either way (ahead on the K-quant files and BitNet, 10% behind on Qwen3-0.6B); prompts 3-9% behind on the small models, level on the rest. The batched matmul runs at 85-90% of the GPU's measured matrix peak.
