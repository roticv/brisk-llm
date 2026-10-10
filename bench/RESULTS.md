# Benchmark results

Raw output is in `bench/results/`. Reproduce with `bench/run_llama_bench.sh` and `build/membw`.

## MacBook Air M4 (4 performance + 6 efficiency cores, 32 GB)

llama.cpp commit `8216c84` (2026-10-05), 512-token prompt, 128 generated tokens, 5 repetitions, 20 s cool-down between tests. Speeds in tokens/s.

| Model | Backend | Threads | Prompt | Generation |
|---|---|---:|---:|---:|
| Qwen3-0.6B Q4_0 | CPU | 4 | 934 | 167 |
| Qwen3-0.6B Q4_0 | CPU + KleidiAI | 4 | 1002 | 166 |
| Qwen3-0.6B Q4_0 | CPU | 10 | 311 ± 69 | 129 ± 59 |
| Qwen3-0.6B Q4_0 | Metal | 4 | 2728 | 175 |
| Qwen3-1.7B Q4_0 | CPU | 4 | 354 | 64.7 |
| Qwen3-1.7B Q4_0 | CPU + KleidiAI | 4 | 390 | 63.6 |
| Qwen3-1.7B Q4_0 | CPU | 10 | 401 | 55.7 |
| Qwen3-1.7B Q4_0 | Metal | 4 | 946 | 78.3 |
| Qwen3-1.7B Q4_K_M | CPU | 4 | 224 | 58.9 |
| Qwen3-1.7B Q4_K_M | CPU | 10 | 301 | 53.8 |
| Qwen3-1.7B Q4_K_M | Metal | 4 | 893 | 74.8 |

"CPU + KleidiAI" is llama.cpp built with `-DGGML_CPU_KLEIDIAI=ON` (Arm's SME kernels).

### Memory bandwidth and the generation ceiling

Sequential read bandwidth from `membw`: 47 GB/s on 1 thread, 78 GB/s on 4, 110 GB/s on 10.

Ceiling = bandwidth / model file size. This is approximate: `membw` is a simple summing loop, so the true peak may be somewhat higher.

| Model | Size | Ceiling, 4 threads | Ceiling, 10 threads | llama.cpp CPU best | Share of 4-thread ceiling |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B Q4_0 | 382 MB | 205 | 289 | 167 | 81% |
| Qwen3-1.7B Q4_0 | 1057 MB | 74 | 104 | 64.7 | 87% |
| Qwen3-1.7B Q4_K_M | 1107 MB | 71 | 100 | 58.9 | 83% |

### What this means

- Generation on the performance cores is already close to the bandwidth limit, so better kernels alone can gain roughly 15-25% there.
- The efficiency cores add about 40% more bandwidth, but llama.cpp gets slower when given all 10 cores. Scheduling work across both core types in proportion to their speed is the largest generation opportunity at 4 bits.
- Prompt processing is compute-bound: Metal is 2.5-4x faster than CPU, and KleidiAI adds only about 10%. There is room for a better CPU path.
- Beyond that, fewer bits per weight is the remaining lever, as the roadmap assumes.

### brisk on the M4 (current)

Idle machine, 20 s pause between runs (as llama-bench's --delay; the fanless Air throttles otherwise, and brisk's compute-heavier kernels suffer more from that than llama.cpp's), prompt = ~500 tokens, generation = 128 tokens from an almost empty context, medians of 5. llama.cpp CPU and Metal numbers are from the baseline runs on the same files. Tokens/s.

| File | Threads | brisk prompt | llama.cpp CPU | llama.cpp Metal | brisk generation | llama.cpp CPU | llama.cpp Metal |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B pure Q4_0 | 4 | 687 | 697 | 2845 | 178 | 189 | 198 |
| Qwen3-0.6B pure Q4_0 | 10 | 1043 | 882 | 2759 | 219 | 162 | 165 |
| Qwen3-1.7B pure Q4_0 | 4 | 269 | 347 | 1003 | 69 | 71 | 86 |
| Qwen3-1.7B pure Q4_0 | 10 | 376 | 426 | 1015 | 79 | 66 | 69 |
| Qwen3-1.7B Q4_K_M | 4 | 247 | 224 | 893 | 59 | 59 | 75 |
| Qwen3-1.7B Q4_K_M | 10 | 376 | 301 | 897 | 76 | 54 | 65 |
| Qwen3-1.7B Q4_0 (Q6_K embedding, Q4_1) | 4 | 269 | 353 | 946 | 63 | 65 | 78 |
| Qwen3-1.7B Q4_0 (Q6_K embedding, Q4_1) | 10 | 358 | 401 | 958 | 81 | 56 | 74 |
| Qwen3-0.6B Q4_0 (Q6_K embedding) | 4 | 644 | 934 | 2728 | 162 | 167 | 175 |
| Qwen3-0.6B Q4_0 (Q6_K embedding) | 10 | 844 | 311 | 2692 | 181 | 129 | 159 |
| BitNet 2B4T TQ2_0 (Q6_K embedding) | 4 | 209 | 139 | | 71 | 30 | |
| BitNet 2B4T TQ2_0 (Q6_K embedding) | 10 | 368 | 81 | | 99 | | |
| BitNet 2B4T TQ2_0 (Q4_0 embedding) | 4 | 240 | 130 | | 79 | 36 | |
| BitNet 2B4T TQ2_0 (Q4_0 embedding) | 10 | 365 | 125 | | 108 | 10 | |

Reading: generation is level with llama.cpp's CPU path at 4 threads (within 3-6%) and 18-45% ahead at 10, where it also passes Metal; prompt processing is 7-23% behind on Q4_0 at 4 threads, ahead on Q4_K_M, and ahead at 10 threads on the small model. All formats now use the four-row interleaved layouts with smmla tiles (Q4_0, Q4_K, Q6_K, TQ2_0) on CPUs with i8mm. Earlier tables in this file's history were measured without pauses and understated brisk, most of all on BitNet (35-40 then, 71-108 now).

Per token, the BitNet files read 0.72-0.81 GB, less than Qwen3-1.7B Q4_0 (0.97 GB) but not by the 3x the "0.4 GB" figure suggests, because the 128k-vocabulary output layer is a third of the bytes. Perplexity matches llama.cpp within 0.13%; the Llama 3 tokenizer matches on 2,117 test strings.

### Speculative decoding on the M4 (Phase 3)

Qwen3-1.7B pure Q4_0, 4 threads, greedy, output identical to plain decoding:

| Text | plain | draft = Qwen3-0.6B | lookup drafting |
|---|---:|---:|---:|
| fresh answer (200 tokens) | 70.5 | 42.1 (55% of guesses accepted) | 44.0 (12%) |
| "repeat the text above" (200 tokens) | 56.9 | | 107 (5 guesses), 135 (12 guesses), 99% accepted |

A draft one third the target's cost loses more than it saves; lookup drafting is free and wins only where the output repeats the context.

## Raspberry Pi 4 Model B (4 GB, Cortex-A72 x4, Debian 13, 64-bit)

Same llama.cpp commit `8216c84`, CPU only, 4 threads, 5 repetitions, 20 s cool-down. Speeds in tokens/s.

| Model | Prompt | Generation |
|---|---:|---:|
| Qwen3-0.6B Q4_0 | 19.5 | 9.16 |
| Qwen3-1.7B Q4_0 | 6.57 | 3.36 |
| Qwen3-1.7B Q4_K_M | 6.04 | 3.25 |

The firmware reported `throttled=0xe0000` after the run (frequency capping and the soft temperature limit occurred during it; 67 C at the end), so prompt-processing numbers may be slightly pessimistic. Generation is bandwidth-bound and unaffected by the clock.

### Memory bandwidth and the generation ceiling

Sequential read bandwidth from `membw`: 4.3 GB/s on 1 thread, and no higher with 2, 3 or 4 threads (4.3, 4.1, 4.0). A single core saturates the memory.

| Model | Size | Ceiling | llama.cpp | Share of ceiling |
|---|---:|---:|---:|---:|
| Qwen3-0.6B Q4_0 | 382 MB | 11.3 | 9.16 | 81% |
| Qwen3-1.7B Q4_0 | 1057 MB | 4.1 | 3.36 | 83% |
| Qwen3-1.7B Q4_K_M | 1107 MB | 3.9 | 3.25 | 83% |

### brisk on the Pi (Phase 2)

Same files and settings, 4 threads. Prompt = ~500-token prompt; generation = 128 tokens from an almost empty context. llama.cpp numbers for the pure files are from `raspberrypi-llama.cpp-8216c84-pure.md`.

| Model | brisk prompt | llama.cpp prompt | brisk generation | llama.cpp generation |
|---|---:|---:|---:|---:|
| Qwen3-0.6B pure Q4_0 | 19.4 | 19.0 | 9.9 | 9.7 |
| Qwen3-1.7B pure Q4_0 | 6.6 | 5.8 | 3.7 | 3.3 |
| Qwen3-0.6B Q4_0 (Q6_K embedding) | | 19.5 | 8.8 | 9.2 |
| Qwen3-1.7B Q4_0 (Q6_K embedding) | | 6.6 | 3.4 | 3.4 |
| Qwen3-1.7B Q4_K_M | | 6.0 | 3.4 | 3.3 |

The Pi reached 82 C and its soft temperature limit during this run, so the prompt numbers may be slightly pessimistic.

### BitNet b1.58 2B4T on the Pi (Phase 3)

Same files as on the Mac, 4 threads, short runs (the Pi reaches its soft temperature limit during longer ones).

| File | brisk generation | llama.cpp generation |
|---|---:|---:|
| TQ2_0 + Q6_K embedding (0.81 GB/token) | 4.3 | |
| TQ2_0 + Q4_0 embedding (0.72 GB/token) | 4.6 | 4.7 |

For comparison Qwen3-1.7B Q4_0 (0.97 GB/token) generates at 3.6. The bandwidth ceiling for the 0.72 GB file is 6.0 tokens/s; the ternary kernel is also close to compute-bound on the A72 (about 3 multiply-adds per instruction), so cutting bytes further would need a lookup-table kernel to pay off.

### What this means

- Generation is bandwidth-bound, as the user's earlier benchmarks also showed. Better kernels on the same file can gain at most about 20%.
- The only large lever is bytes per token: fewer bits per weight (a 0.4 GB BitNet model has a ceiling near 10 tokens/s, three times Qwen3-1.7B at 4 bits) and skipping most of the output layer.
- Prompt processing is compute-bound here (19.5 tokens/s on the 0.6B against a 11.3 tokens/s generation ceiling means weights are reused across the batch), so kernel work still matters for prompts.
