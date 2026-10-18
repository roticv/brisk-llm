# brisk-llm roadmap

## Goal

An LLM inference engine in C++20 that is faster than llama.cpp on two machines:

- MacBook Air M4 (AArch64, NEON, `dotprod`, `i8mm`, SME2, Metal GPU, fanless; 4 performance + 6 efficiency cores)
- Raspberry Pi 4 Model B, 4 GB (Cortex-A72 x4, baseline NEON only, 4.3 GB/s memory bandwidth)

Scope started narrow: two models, CPU first, one codebase for both machines. The CPU work (Phases 0-4) and the Metal backend (Phase 5) are done; Phase 6 widens the model set.

| Model | Role | Baseline to beat |
|---|---|---|
| Qwen3-1.7B, 4-bit (Qwen3-0.6B for development) | Control: like-for-like comparison and correctness reference | llama.cpp |
| BitNet b1.58 2B4T, ternary | Headline speed target | bitnet.cpp and llama.cpp |

## What the baselines showed

Full numbers are in `bench/RESULTS.md`. The findings that shape the plan:

- **Generation is memory-bandwidth-bound on both machines.** llama.cpp reaches 81-87% of the ceiling (bandwidth divided by model bytes) on the M4's performance cores and 81-83% on the Pi. Faster kernels on the same 4-bit file can gain at most about 20%.
- **The Pi's bandwidth is saturated by one core.** Threads do not help generation there; only reading fewer bytes per token does.
- **The M4's efficiency cores add about 40% more bandwidth**, but llama.cpp gets slower when given all 10 cores. Using both core types well is the largest 4-bit gain on the Mac.
- **Prompt processing is compute-bound on both machines**, so kernel quality still matters there. On the Mac, llama.cpp's Metal backend is 2.5-4x faster than its CPU path.

So the plan reaches llama.cpp's level on 4-bit models first, then spends its effort on bytes per token rather than on kernel tuning.

## Metrics

Measured on both machines, for every phase:

- Generation speed (tokens/s)
- Prompt processing speed (tokens/s)
- Time to first token
- Load time and peak memory

Rules: same model file, same prompt, same thread count as the baseline. On the Air, report sustained speed after thermal throttling, not just the first seconds. On the Pi, record `vcgencmd get_throttled` after each run.

## Phase 0: Baselines and harness — done

- llama.cpp built at the same commit on both machines; benchmark script and bandwidth tool written (`bench/`).
- Results and ceilings recorded in `bench/RESULTS.md`.
- bitnet.cpp is not built yet; it is only needed as the Phase 3 baseline.

## Phase 1: Correct scalar engine — done

- CMake project, ASan/UBSan builds, bounds-checked GGUF loader over mmap, Qwen tokenizer, float32 Qwen3 forward pass, sampler, `brisk` CLI.
- Tokenizer matches llama.cpp on 2,060 test strings; greedy output matches token for token on eight prompts with logits within 0.04; float32 perplexity matches to four decimals.
- Not done: cross-compilation to the Pi from the Mac. The Pi builds natively with its own GCC for now.

## Phase 2: Reach llama.cpp on 4-bit models — done for Q4_0

- Done: Q4_0, Q4_1, Q4_K and Q6_K weights with 8-bit activations; NEON kernels (baseline and `dotprod`) chosen at startup; thread pool with dynamic chunks; batched prompt processing; GGUF parser fuzzer (`tests/gguf_fuzz.cpp`); `brisk perplexity` and `tests/compare_perplexity.py` as the correctness check for quantised models (per-logit comparison is chaotic for 8-bit arithmetic, so perplexity is the yardstick: brisk matches llama.cpp within about 1%).
- Pi results: generation at or slightly above llama.cpp on every file; prompt processing within 12% on the 0.6B and 5% on the 1.7B (`bench/RESULTS.md`).
- Mac results (`bench/RESULTS.md`): with `i8mm` (smmla) batched kernels, a four-row interleaved Q4_0 layout repacked at load, and NEON attention, Qwen3-1.7B generation is 7% behind llama.cpp's CPU path at 4 threads and 21% ahead at 10 (80 vs 66, also ahead of Metal's 69); prompt processing is 30% behind at 4 threads and 16% behind at 10.
- Done later (2026-10-09): the thread pool ends a dispatch when its tasks are done rather than when every worker reports in (the efficiency cores respond slowly, which made each dispatch cost 18-33 us at 10 threads), and Q/K/V and gate/up are single dispatches; 10-thread generation gained 5-10%.
- Done later (2026-10-10): Q4_K and Q6_K got the four-row interleaved layout, sdot generation kernels and smmla tiles, and Q4_1 NEON kernels; Q4_K_M generation went from 50 to 59 (4 threads) and 76 (10 threads) against llama.cpp's 59 / 54, and prompts from 65 to 247. The same Q6_K kernels doubled BitNet's prompt rate (its output layer is Q6_K).

**Done when:** Qwen3-1.7B generation is within 10% of llama.cpp on both machines and prompt processing is within 25%, with perplexity matching. On the Mac, generation with all 10 cores should beat llama.cpp's 4-core number. Met on the Pi and, for Q4_0, on the Mac except prompt processing at 4 threads (30% behind).

## Phase 3: Fewer bytes per token — stopped after BitNet and speculative decoding

Status so far (details in `bench/RESULTS.md`):

- **BitNet b1.58 2B4T runs in brisk** (Llama 3 tokenizer, squared-ReLU feed-forward, sub-norms, TQ2_0 ternary kernels with a four-row interleaved layout and an smmla prompt tile). Perplexity matches llama.cpp within 0.13%. The baseline is mainline llama.cpp with a one-line local patch (its BitNet graph uses SiLU; this model uses squared ReLU); bitnet.cpp's generic ternary kernel has no NEON path and its ARM path needs a different conversion and generated kernels.
- **The "0.4 GB" model reads 0.72-0.81 GB per token**, because its 128k-token embedding/output layer is 185-269 MB on its own. That is 25% fewer bytes than Qwen3-1.7B Q4_0, not 3x. Vocabulary pruning and the output-layer shortcut are therefore part of the BitNet work, not separate from it.
- **On the M4 ternary pays only with cooling pauses:** generation is compute-bound (four times the work per byte of Q4_0), so back-to-back runs on the fanless Air throttle to 35-40 tokens/s; with 20 s pauses between runs it reaches 79-108, above Qwen3-1.7B Q4_0's 69-79, and 2-3x llama.cpp on the same file.
- **On the Pi it does, modestly:** 4.6 tokens/s against 3.6 for Qwen3-1.7B Q4_0, for a model with 40% more parameters.
- **Speculative decoding is built and exact** (`--draft lookup` or `--draft <model>`), but with Qwen3-0.6B drafting the 1.7B it is slower than plain decoding (42 vs 70 tokens/s): the draft costs a third of a target token per guess. Lookup drafting is free and gives 2-2.4x on text that repeats its context, nothing on fresh text. Model drafting needs a draft under a tenth of the target's cost, which no Qwen3 pair offers.

- **The remaining levers have small ceilings on the Pi.** BitNet reads 0.72 GB per token: 537 MB ternary layers, 185 MB output layer. Vocabulary pruning with an ASCII-only rule keeps 76% of the tokens (1.07-1.10x); pruning to ~32k tokens or an exact output-layer shortcut tops out at 1.35x; 1.6-bit ternary packing at 1.15x; all together about 1.6x, and the ternary kernel would need to become a lookup-table kernel for that to materialise, since the A72 is nearly compute-bound on it. On the Mac, ternary is a loss and the 4-bit Qwen3 path stays the best option.

Decision (2026-10-07): stop here and consolidate rather than pursue those levers. The items below are not started.

- Vocabulary pruning, the output-layer shortcut, 1.6-bit ternary packing, lookup-table ternary kernels (T-MAC style), sequential layout and prefetching.


This is the phase that can move the ceiling, so it comes before any kernel polishing.

- **BitNet b1.58 2B4T:** load the model; add ternary linear layers, 8-bit activations and the squared-ReLU feed-forward block. Pack ternary weights densely (2 bits per weight first, then about 1.6). Lookup-table kernels using NEON `tbl`, which need only baseline NEON and so run fully on the Pi 4. Build bitnet.cpp as the baseline and verify outputs against it.
- **Output-layer shortcut:** the vocabulary projection is 18-26% of the weight bytes in Qwen3. Shortlist candidate tokens with a cheap pass, then compute exact logits only for them. Must keep greedy output identical on the test prompts, and must be switchable off.
- **Vocabulary pruning:** a tool that drops never-used tokens from the embedding and output tables and writes a smaller GGUF (Qwen3's 151,936-token vocabulary is mostly unused for English and code; keeping about 50k rows cuts the output layer by roughly two thirds). The tokenizer must map pruned tokens to their byte fallbacks so any input still tokenizes. Output is unchanged whenever the model would not have produced a pruned token.
- **Speculative decoding:** draft several tokens with Qwen3-0.6B (or from n-grams in the prompt, which needs no draft model), then verify them in one batched pass of the main model. Output is identical to the main model alone; the gain depends on how often the draft is right. Needs the batched forward pass from Phase 2.
- **Sequential layout and prefetching**, so each pass reads the weights at the measured bandwidth.

**Done when:** BitNet outputs match bitnet.cpp and generation beats bitnet.cpp and llama.cpp on both machines; the output-layer shortcut, vocabulary pruning and speculative decoding each give a measured generation gain on Qwen3-1.7B on both machines with unchanged greedy output on the test prompts.

## Phase 4: Prompt processing and remaining 4-bit headroom — done, criterion partly met

- Done: fused dispatch (Q/K/V and gate/up share one activation quantisation and one dispatch), K-quant grouped tiles (Q4_Kx4, Q6_Kx4 with smmla), strided kernel output straight into the attention layout, row-band blocking for the NEON and i8mm batched kernels, head-major K/V cache with NEON attention, and no allocation during generation. Unit tests, pinned perplexity and CI came with it.
- Outcome on Qwen3-1.7B Q4_0: Pi prompt processing 1.9x llama.cpp and generation 12% ahead; M4 generation 8% ahead at 4 threads and 40% at 10; M4 prompt processing 6% behind at 4 threads and 12% ahead at 10.

Dropped: A72-specific kernel tuning for generation. The Pi is bandwidth-bound, so it cannot pay off.

**Done when:** on Qwen3-1.7B, generation is at least 15% faster and prompt processing at least 30% faster than llama.cpp on both machines. Prompt processing: met on the Pi, not on the M4 at 4 threads. Generation: not met at 15% on either machine at 4 threads, and not reachable there by kernel work, since both engines sit at 80-90% of the bandwidth ceiling; met on the M4 with all cores.

## Phase 5: Mac-only acceleration — Metal backend done, at parity

- Metal backend behind `--gpu` (`src/metal/`): an `Engine` interface shared with the CPU `Session`; weights used in place from the memory-mapped file (no-copy buffer), K/V cache on the GPU, one command buffer per batch; shaders embedded as source and compiled at start. Kernels: Q4_0 gemv (two rows per simdgroup, fused q/k/v and gate/up dispatches, epilogues for the residual add and gating), a 64x64 simdgroup-matrix tile for batches (block-major operands, half activations written by the producing kernels, fused epilogues), per-head norm + RoPE + cache write, attention on the matrix unit by blocks of 8 tokens, and a split-position attention for single tokens with a combine pass. `metal_test` checks both matmuls against a float reference; `metal_bench` gives GFLOPS per shape; the perplexity pins hold on the GPU (`check_perplexity.py --gpu`).
- Result on Qwen3-1.7B Q4_0: 940 tokens/s prompt and 89.6 generation (80 after a 500-token prompt) against llama.cpp Metal's 1034 and 88.4; on the 0.6B, 2615 / 183 against 2825 / 198. Prompt processing is 2.9x the CPU path. The tiled matmul runs at 85-90% of the GPU's measured matrix peak and faster than llama.cpp's own kernel in isolation; the remaining 7-9% on prompts is in the smaller passes and launch gaps, and generation is at the bandwidth ceiling for the large matrices. Details in `bench/RESULTS.md`.
- Added since: BitNet (TQ2_0) on the GPU at 634 prompt / 93 generation (llama.cpp Metal 655 / 84), split-position attention for small batches, back-off in the Speculator. Not done: Q6_K/Q8_0 output matrices (the common "Q4_0" downloads have one), K-quant GPU kernels, a half-precision K/V cache, a ternary gemv at bandwidth, SME2 CPU kernels.

**Done when (original):** the Mac numbers beat llama.cpp's best configuration. Reached for generation on the 1.7B (level), not for prompt processing (7-9% behind) or the 0.6B.

## Phase 6: More model families — not started

brisk loads Qwen3 and BitNet b1.58 only. Most of what people run locally is a handful of other families, and most downloads are Q4_K_M files (Q4_K with Q6_K, sometimes Q5_K, for attention V and the output) rather than Q4_0. In order of cost:

1. **Llama 3.x, Mistral, SmolLM2, DeepSeek-R1-distill-Llama** (GGUF architecture `llama`) and **Qwen2.5, R1-distill-Qwen** (`qwen2`) — done. Same forward pass as Qwen3 without the per-head q/k norms; `qwen2` adds q/k/v biases, `llama` uses adjacent-pair RoPE and Llama 3's per-frequency rope factors. Llama-3.2-1B-Instruct and Qwen2.5-1.5B-Instruct (bartowski Q4_K_M) match llama.cpp's tokenizer on all 2,156 test strings and its perplexity within 0.6%; both pinned. The SentencePiece tokenizer of Mistral 7B and the `tekken`/`smollm` pre-tokenizers are not implemented, so those files load but do not tokenize yet.
2. **Q5_K weights** — not needed so far: the Q4_K_M files of Llama 3.2 and Qwen2.5 use only Q4_K and Q6_K. Still open for files that have it. **Q5_0 and Q8_0 weights** — done (for Gemma's downloads), with i8mm tiles and a Q5_0x4 repacked layout.
3. **Q4_K, Q6_K and Q4_1 on the GPU** — done (gemv, tile and embedding for each; dispatches split by format within a layer). The common Q4_0 downloads (Q6_K output, some Q4_1) and the Q4_K_M files run with `--gpu`, checked against the CPU pins and llama.cpp. After tuning the tile staging and the Q4_K gemv, prompts are level with llama.cpp Metal (within 3-4%) and generation 4-8% behind, the rest being small-matrix dispatches (`bench/RESULTS.md`). Q8_0 and Q5_K are not on the GPU.
4. **Gemma 3** (`gemma3`) — done, on the CPU and the GPU: SentencePiece tokenizer (identical to llama.cpp on the test corpus), √dim embedding scaling, Q/K norms, post-norms around attention and the FFN (the GGUF converter folds the `1 + w` into the weights), GELU gating, sliding-window layers (every layer but each sixth; window and bases from the file), optional final softcap, and 256-wide heads in the Metal attention kernels. gemma-3-1b-it as pure Q4_0 matches llama.cpp's perplexity within 0.8% and its greedy output on every test prompt; pinned. The Q4_K_M downloads of the 1B use Q5_0 and Q8_0 (its 1152-wide embedding is not a multiple of 256); both formats were added (CPU: reference, NEON and dotprod; GPU: gemv, tile and embedding), so the stock file loads and matches llama.cpp within 0.15%; with the Q5_0x4 repacked layout and i8mm tiles its CPU prompt processing is 408-482 tokens/s against llama.cpp's 596 (generation 81 vs 73). After tuning (a NEON GELU, a Q4_0 gemv restructured after llama.cpp's) it is level with llama.cpp on generation on both backends and 4-6% behind on prompts (`bench/RESULTS.md`). Untested on the 4B and larger (head dim 256 with 2 query heads per key/value head, and the 27B's different attention scale).
5. **Gemma 4** (`gemma4`, April 2026; E2B, E4B, 12B, 26B-A4B MoE, 31B; ggml-org and unsloth publish Q4_0/Q8_0 GGUFs) — E2B done on the CPU: the `gemma4` tokenizer (BPE by rank over raw UTF-8 characters, newline-run splitting), per-layer head sizes (256 sliding, 512 global) with the file's sliding pattern, K/V sharing (the last 20 layers read layer 13's or 14's cache), V normalised without a weight, no attention scaling, per-layer feed-forward widths, the per-layer embeddings (table row plus projected input, gated into the residual after each layer), the layer output scalars, the softcap. Matches llama.cpp within 0.8% and is level with it in speed (`bench/RESULTS.md`). Not done: the GPU (512-wide heads, sharing, PLE), the MoE 26B-A4B, the dense 12B/31B (untested; no PLE there). Original notes follow. Builds on Gemma 3 and adds, per the config and Google's write-ups: a 4:1 (E2B) or 5:1 local:global layer pattern with the last layer global and sliding windows of 512 (E2B/E4B) or 1024; GQA with 2 query heads per K/V head on local layers and 8 on global ones, head dim 256 local and 512 global, with K = V on global layers; "pruned" RoPE on global layers (positions applied to the first 25% of the pairs only); K/V sharing, where the last 20 of the E2B's 35 layers reuse an earlier layer's K/V instead of projecting their own; per-layer embeddings (the E2B's 2.3B "effective" parameters are 5.1B with them: a 262144 x 256 table per layer, one row read per token and mixed into the residual stream after each block, so they cost disk, not bandwidth, which suits the Pi); the 26B-A4B is a 128-expert MoE with 8 active plus a shared expert. Multimodal inputs go through the same decoder and can be ignored for text. The E2B is the natural Pi 4 model of this family (about 1.3 GB read per token at Q4_0). Another two to three days on top of Gemma 3; worth checking llama.cpp's `gemma4` graph for the exact norm placement and the PLE gating before starting.
6. **Phi-4-mini** (`phi3`), if wanted.

Each family is checked the way Qwen3 was: tokenizer ids against llama.cpp on the test strings, perplexity within 1% on the pinned texts, and a pin added.

**Done when:** the Q4_K_M downloads of Llama 3.x and Qwen2.5 and the Q4_0 files of Gemma 3 and Gemma 4 E2B/E4B load and match llama.cpp's perplexity on both machines, with the GPU path taking their output matrices. Reached for Llama 3.x, Qwen2.5 and Gemma 3 on the Mac (both backends) and Gemma 4 E2B on the CPU; Gemma 4 on the GPU and the Pi runs remain.

## Later, if wanted

- OpenAI-compatible HTTP server.
- Sparsity experiments (skipping inactive neurons).
- Layer pruning and 3-bit weights, which trade some quality for fewer bytes.
- Own weight file format, if GGUF layout becomes the bottleneck.
- Cross-compilation to the Pi from the Mac.

## Where it stands against the goal

Like-for-like on the same files (details and conditions in `bench/RESULTS.md`):

| | brisk | llama.cpp |
|---|---|---|
| M4, Qwen3-1.7B Q4_0, generation 4 / 10 threads | 77 / 93 | 71 / 66 (Metal 86 / 69) |
| M4, Qwen3-1.7B Q4_0, prompt 4 / 10 threads | 325 / 479 | 347 / 426 |
| M4 GPU, Qwen3-1.7B Q4_0, prompt / generation | 940 / 89.6 | Metal 1034 / 88.4 |
| M4 GPU, Qwen3-0.6B Q4_0, prompt / generation | 2615 / 183 | Metal 2825 / 198 |
| M4 GPU, BitNet 2B4T, prompt / generation | 634 / 93 | Metal 655 / 84 |
| M4, Llama-3.2-1B Q4_K_M, CPU 4 threads prompt / generation | 375 / 83 | 321 / 86 |
| M4 GPU, Llama-3.2-1B Q4_K_M, prompt / generation | 1331 / 104 | Metal 1370 / 104 |
| M4, Qwen2.5-1.5B Q4_K_M, CPU 4 threads prompt / generation | 273 / 66 | 285 / 68 |
| M4 GPU, Qwen2.5-1.5B Q4_K_M, prompt / generation | 940 / 76 | Metal 981 / 83 |
| M4, Gemma-3-1B Q4_0, CPU 4 threads prompt / generation | 657 / 124 | 684 / 117 |
| M4 GPU, Gemma-3-1B Q4_0, prompt / generation | 1814 / 128 | Metal 1924 / 127 |
| M4, Gemma-4-E2B Q4_0, CPU 4 threads prompt / generation | 194 / 42.5 | 182 / 41.8 |
| M4, Qwen3-1.7B Q4_K_M, generation 4 / 10 threads | 59 / 76 | 59 / 54 (Metal 75 / 65) |
| M4, Qwen3-1.7B Q4_K_M, prompt 4 / 10 threads | 247 / 376 | 224 / 301 |
| M4, Qwen3-0.6B Q4_0, generation 4 / 10 threads | 178 / 219 | 189 / 162 (Metal 198 / 165) |
| M4, BitNet 2B4T, generation 4 / 10 threads | 79 / 108 | 36 / 10 |
| Pi 4, Qwen3-1.7B Q4_0, prompt / generation | 11.0 / 3.7 | 5.8 / 3.3 |
| Pi 4, Qwen3-0.6B Q4_0, prompt / generation | 19.4 / 9.9 | 19.0 / 9.7 |
| Pi 4, BitNet 2B4T, generation | 4.6 | 4.7 |

Faster than llama.cpp: generation on the Pi, prompt processing on the Pi (1.9x on the 1.7B), generation on the M4 (8% ahead at 4 threads, 40% at 10, also past Metal), BitNet on the M4 (2-3x), Q4_K_M files on the M4, M4 prompt processing at 10 threads, lookup-drafted generation on repetitive text (2x). Not faster: M4 prompt processing on Q4_0 at 4 threads (6% behind), and on the GPU, where brisk's Metal backend is level with llama.cpp's on 1.7B generation, 11% ahead on BitNet generation, and 3-9% behind on prompts. Both engines sit at 80-90% of the memory-bandwidth ceiling for generation, which is why the generation margins are small.

## What's next

Phase 6 (model families) is the open work. BitNet runs on the GPU (TQ2_0 gemv and tile): generation 93 tokens/s against llama.cpp Metal's 84 (the premise that llama.cpp lacked ternary Metal kernels was wrong; it has them), prompt 634 against 655; brisk's CPU path with all cores (102) is still slightly ahead of its GPU for generation. The ternary gemv reads weights at 90 GB/s of the 120 ceiling, the decode being the rest. Speculative decoding on the GPU is measured (`bench/RESULTS.md`): 2.9x on repeated text with 24 lookup guesses, a 26% loss on fresh text even with the new back-off, because a verify batch of n tokens costs about 0.7 n single tokens on the GPU; a batched gemv that reads the weights once per batch (like llama.cpp's `mul_mv_mma`) would fix that and five attempts at it were slower than the plain gemv. Also open: Q6_K/Q8_0 output matrices on the GPU, a half-precision K/V cache, SME2 CPU kernels.

## Open questions

- The GPU is now in scope and the Metal backend is at parity with llama.cpp's. Beating it there by a clear margin would have to come from elsewhere than the matmul (already at 85-90% of the GPU's peak): speculative decoding on the GPU, or ternary (BitNet) GPU kernels where the weights are 3x smaller.
- The Pi throttles under sustained load (soft temperature limit reached during benchmark runs). Cooling would make its prompt numbers more reliable.
