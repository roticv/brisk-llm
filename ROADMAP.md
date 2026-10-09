# brisk-llm roadmap

## Goal

An LLM inference engine in C++20 that is faster than llama.cpp on two machines:

- MacBook Air M4 (AArch64, NEON, `dotprod`, `i8mm`, SME2, Metal GPU, fanless; 4 performance + 6 efficiency cores)
- Raspberry Pi 4 Model B, 4 GB (Cortex-A72 x4, baseline NEON only, 4.3 GB/s memory bandwidth)

Scope is deliberately narrow: two models, CPU first, one codebase for both machines.

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
- Not done: the K-quants (Q4_K, Q6_K) have no batched tiles or interleaved layout, so Q4_K_M files trail llama.cpp on the Mac.

**Done when:** Qwen3-1.7B generation is within 10% of llama.cpp on both machines and prompt processing is within 25%, with perplexity matching. On the Mac, generation with all 10 cores should beat llama.cpp's 4-core number. Met on the Pi and, for Q4_0, on the Mac except prompt processing at 4 threads (30% behind).

## Phase 3: Fewer bytes per token — stopped after BitNet and speculative decoding

Status so far (details in `bench/RESULTS.md`):

- **BitNet b1.58 2B4T runs in brisk** (Llama 3 tokenizer, squared-ReLU feed-forward, sub-norms, TQ2_0 ternary kernels with a four-row interleaved layout and an smmla prompt tile). Perplexity matches llama.cpp within 0.13%. The baseline is mainline llama.cpp with a one-line local patch (its BitNet graph uses SiLU; this model uses squared ReLU); bitnet.cpp's generic ternary kernel has no NEON path and its ARM path needs a different conversion and generated kernels.
- **The "0.4 GB" model reads 0.72-0.81 GB per token**, because its 128k-token embedding/output layer is 185-269 MB on its own. That is 25% fewer bytes than Qwen3-1.7B Q4_0, not 3x. Vocabulary pruning and the output-layer shortcut are therefore part of the BitNet work, not separate from it.
- **On the M4 ternary does not pay:** generation is compute-bound (four times the work per byte of Q4_0) and the fanless Air throttles, so sustained generation is 35-40 tokens/s against 66 for Qwen3-1.7B Q4_0. brisk is still ahead of llama.cpp on the same file (30-36, with large variance).
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

## Phase 4: Prompt processing and remaining 4-bit headroom — not started

- Cache-blocked quantised matrix multiply for prompt processing.
- Static memory planning, fused operations, no allocation during generation.
- Anything the profiles still show, now that the big levers are in.

Dropped: A72-specific kernel tuning for generation. The Pi is bandwidth-bound, so it cannot pay off.

**Done when:** on Qwen3-1.7B, generation is at least 15% faster and prompt processing at least 30% faster than llama.cpp on both machines.

## Phase 5: Mac-only acceleration — not started

Decide from measurements which of these is worth doing:

- SME2 kernels for prompt processing (llama.cpp's KleidiAI path gains only about 10%, so there may be more available).
- A Metal backend with fused kernels.

**Done when:** the Mac numbers beat llama.cpp's best configuration (CPU or Metal, whichever is faster), or the measurements show the CPU path already does.

## Later, if wanted

- OpenAI-compatible HTTP server.
- More model families (Gemma, Llama).
- Sparsity experiments (skipping inactive neurons).
- Layer pruning and 3-bit weights, which trade some quality for fewer bytes.
- Own weight file format, if GGUF layout becomes the bottleneck.
- Cross-compilation to the Pi from the Mac.

## Where it stands against the goal

Like-for-like on the same files (details and conditions in `bench/RESULTS.md`):

| | brisk | llama.cpp |
|---|---|---|
| M4, Qwen3-1.7B Q4_0, generation 4 / 10 threads | 67 / 83 | 71 / 66 (Metal 86 / 69) |
| M4, Qwen3-1.7B Q4_0, prompt 4 / 10 threads | 247 / 373 | 347 / 426 |
| M4, Qwen3-1.7B Q4_K_M, generation 4 / 10 threads | 50 / 51 | 59 / 54 |
| M4, BitNet 2B4T, generation 4 threads | 40 | 36 |
| Pi 4, Qwen3-1.7B Q4_0, prompt / generation | 6.6 / 3.7 | 5.8 / 3.3 |
| Pi 4, Qwen3-0.6B Q4_0, prompt / generation | 19.4 / 9.9 | 19.0 / 9.7 |
| Pi 4, BitNet 2B4T, generation | 4.6 | 4.7 |

Faster than llama.cpp: generation on the Pi, generation on the M4 with all cores, BitNet on the M4, lookup-drafted generation on repetitive text (2x). Not faster: M4 generation on 4 threads (7% behind), M4 prompt processing (16-30% behind), K-quant files on the M4. Both engines sit at 80-90% of the memory-bandwidth ceiling for generation, which is why the margins either way are small.

## Open questions

- Is the Mac GPU in scope? Metal is the only path to substantially faster prompt processing on the Mac.
- The Pi throttles under sustained load (soft temperature limit reached during benchmark runs). Cooling would make its prompt numbers more reliable.
