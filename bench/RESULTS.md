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

### brisk on the M4 (Phase 2)

Idle machine, pure Q4_0 files for both engines (llama.cpp numbers from `MacBook-Air-llama.cpp-8216c84-pure.md`), prompt = ~500 tokens, generation = 128 tokens from an almost empty context, medians of 5. Tokens/s.

| Model | Threads | brisk prompt | llama.cpp CPU prompt | llama.cpp Metal prompt | brisk generation | llama.cpp CPU generation | llama.cpp Metal generation |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B pure Q4_0 | 4 | 294 | 697 | 2845 | 169 | 189 | 198 |
| Qwen3-0.6B pure Q4_0 | 10 | 527 | 882 | 2759 | 151 | 162 | 165 |
| Qwen3-1.7B pure Q4_0 | 4 | 103 | 347 | 1003 | 62 | 71 | 86 |
| Qwen3-1.7B pure Q4_0 | 10 | 188 | 426 | 1015 | 73 | 66 | 69 |
| Qwen3-1.7B Q4_K_M | 4 | 64 | 224 | 893 | 52 | 59 | 75 |
| Qwen3-1.7B Q4_K_M | 10 | 117 | 301 | 897 | 60 | 54 | 65 |

(Q4_K_M llama.cpp numbers are from the first baseline run on the same file.)

Reading: generation is 10-12% behind llama.cpp's CPU path at 4 threads and ahead of its 10-thread result; prompt processing is 2.3-3.4x behind the CPU path, which uses `i8mm` matrix-multiply kernels (smmla) that brisk does not have yet. The earlier short-run numbers (173 and 67 tokens/s) were taken against the larger mixed files and are superseded by this table.

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
| Qwen3-0.6B pure Q4_0 | 16.8 | 19.0 | 10.0 | 9.7 |
| Qwen3-1.7B pure Q4_0 | 5.5 | 5.8 | 3.7 | 3.3 |
| Qwen3-0.6B Q4_0 (Q6_K embedding) | | 19.5 | 8.8 | 9.2 |
| Qwen3-1.7B Q4_0 (Q6_K embedding) | | 6.6 | 3.4 | 3.4 |
| Qwen3-1.7B Q4_K_M | | 6.0 | 3.4 | 3.3 |

Generation after a ~500-token prompt is lower (7.0 and 3.2 tokens/s on the pure files) because attention over the cache is still scalar code.

### What this means

- Generation is bandwidth-bound, as the user's earlier benchmarks also showed. Better kernels on the same file can gain at most about 20%.
- The only large lever is bytes per token: fewer bits per weight (a 0.4 GB BitNet model has a ceiling near 10 tokens/s, three times Qwen3-1.7B at 4 bits) and skipping most of the output layer.
- Prompt processing is compute-bound here (19.5 tokens/s on the 0.6B against a 11.3 tokens/s generation ceiling means weights are reused across the batch), so kernel work still matters for prompts.
