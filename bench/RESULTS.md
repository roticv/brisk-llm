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

### Metal backend on the M4 (Phase 5)

`--gpu --warmup` against `llama-bench` (which also warms up), run alternately on the same files with 15-20 s pauses, medians of 3. Prompt = 497 tokens (llama-bench: 512); generation = 64 tokens from an almost empty context. Tokens/s.

| File | brisk GPU prompt | llama.cpp Metal | brisk GPU generation | llama.cpp Metal |
|---|---|---|---|---|
| Qwen3-0.6B Q4_0 | 2615 | 2825 | 183 | 198 |
| Qwen3-1.7B Q4_0 | 940 | 1034 | 89.6 | 88.4 |

Generation after a 497-token prompt (context the `tg64` numbers do not have): 80 tokens/s on the 1.7B.

Where the time goes on the 1.7B prompt (GPU time of the batch, 0.53 s): the tiled matmul takes about 0.47 s at 2.8-3.0 TFLOPS, which is 85-90% of the 3.25 TFLOPS that a register-only `simdgroup_multiply_accumulate` loop reaches on this GPU (`bench/metal_bench`; llama.cpp's own matmul measures 2.6 TFLOPS in its `test-backend-ops perf`); attention is 0.03 s; the rest is small. So the 9% gap to llama.cpp on prompt processing is not in the matmul kernel but spread over the remaining passes and launch gaps. In generation the layer gemvs read the weights at ~90 GB/s and the large output matrix at 117 GB/s (of 120 measured by `membw`), so the per-layer dispatches are what costs; the 0.6B, with its smaller matrices, loses more to that. First-run numbers are 10-15% lower than warmed ones: the GPU clock ramps and the mapped weights are touched for the first time.

#### BitNet b1.58 2B4T on the GPU

The TQ2_0 file (e4: Q4_0 embedding and output) with `--gpu`, paced as above, medians of 3, against llama.cpp's Metal backend, which does have TQ2_0 kernels, and brisk's own CPU path. Tokens/s.

| | prompt (498 / 512 tokens) | generation (64, short context) |
|---|---:|---:|
| brisk GPU | 634 | 93 |
| llama.cpp Metal | 655 | 84 |
| brisk CPU, 4 / 10 threads | | 67 / 102 |

Perplexity on the GPU matches the pin to 0.02% through the tile (batches) and 0.19% through the gemv (`BRISK_EVAL_CHUNK=1`, the half-precision decode's rounding).

How the ternary gemv got from 42 to 90 GB/s, measured in isolation on the 6912x2560 gate matrix (`metal_gemv_tq2_0_seconds`): the first kernel (two rows per simdgroup, a quarter block per lane, integer shift-and-mask decode) read weights at 42 GB/s; swapping the decode for llama.cpp's float one (`floor(v / 4^l)` of the byte with telescoped activation coefficients) alone gave 50; copying llama.cpp's structure (four rows per simdgroup, two simdgroups per threadgroup, four blocks in flight with eight lanes each, coefficients shared across the rows) gave 77-82 and generation parity with llama.cpp; the same kernel with the decode removed reads at 121-131 GB/s, so the decode was the remaining cost; running the decode in half precision (exact for bytes and their floor divisions; coefficients scaled by 1/64 to stay in range, since the first attempt overflowed to inf on a long generation while the 64-token perplexity check passed) with eight rows per simdgroup reached 86-91 GB/s. Vectorising the float decode across the lane's four bytes changed nothing (the compiler had done it), and eight rows without the half decode was slower than four.

#### Llama 3.2 and Qwen2.5 (Phase 6)

bartowski's Q4_K_M files (Q4_K with Q6_K attention V and output), idle machine, 12-15 s pauses, two runs each shown; llama.cpp CPU at 4 threads and Metal from `llama-bench` alongside. Prompt = 497 tokens (llama-bench: 512), generation = 64 tokens from an almost empty context. Tokens/s.

| | brisk CPU 4 threads | llama.cpp CPU 4 threads | brisk GPU | llama.cpp Metal |
|---|---:|---:|---:|---:|
| Llama-3.2-1B-Instruct, prompt | 352-396 | 321 | 1160 | 1370 |
| Llama-3.2-1B-Instruct, generation | 82-84 | 86 | 94-101 | 104 |
| Qwen2.5-1.5B-Instruct, prompt | 257-289 | 285 | 813-846 | 981 |
| Qwen2.5-1.5B-Instruct, generation | 66 | 68 | 72-77 | 83 |

On the CPU the new families land where Qwen3 did: generation within 3-5% of llama.cpp at 4 threads (both at the bandwidth ceiling), prompt processing level to 20% ahead.

The GPU numbers above were the first K-quant kernels, 14-15% behind llama.cpp Metal on prompts and 7-13% on generation. Two changes since: the tile's K-quant staging now produces exactly the thread's 16 values in half (it dequantised a whole 32-value sub-block in float and kept half), which took the Q4_K tile from 2523 to 3013 GFLOPS in isolation (Q4_0: 3030) and Q6_K from 2141 to 2792; and the Q4_K gemv applies the sub-block's scale and minimum once per sub-block (sum of q x and sum of x) instead of per value, since it was ALU-bound (slower per byte than Q6_K). After them, paced: Llama-3.2-1B prompt 1331 (llama.cpp Metal 1370), generation 99-101 (104); Qwen2.5-1.5B prompt 900-985 (981), generation 76.5 (83). Qwen2.5's remaining gap is in its small projections (q/k/v and o at 1536 columns, where the gemv reaches only 40-85 GB/s against 90+ on the large ones; a 256-row V matrix is 32 threadgroups) and would need a different split of the work for small matrices.

#### Gemma 3 1B (Phase 6)

gemma-3-1b-it, pure Q4_0 made from bartowski's bf16 with `llama-quantize --pure` (the Q4_K_M downloads use Q5_0 and Q8_0, because the 1152-wide embedding is not a multiple of 256, and brisk has neither). Same conditions as the table above. Tokens/s.

| | brisk CPU 4 threads | llama.cpp CPU 4 threads | brisk GPU | llama.cpp Metal |
|---|---:|---:|---:|---:|
| prompt | 483-562 | 684 | 1700 | 1924 |
| generation | 104 | 117 | 102-106 | 127 |

Those were the first numbers: correct (tokenizer identical on 2,160 strings; perplexity within 0.8% of llama.cpp on the CPU and 0.3% of that on the GPU; greedy output identical to llama.cpp on every test prompt with the F32 file) but 10-20% behind llama.cpp in every cell. Profiling found two causes, neither the attention: on the CPU the GELU gate was a scalar `std::tanh` loop costing 1.2 ms of a 9.6 ms token, now a NEON loop on the polynomial exp (SiLU got the same); on the GPU the Q4_0 gemv reached only 70 GB/s on Gemma's 1152-column matrices, whose 36 blocks per row left most lanes idle on the second pass. The gemv is now structured after llama.cpp's `mul_vec_q_n_f32`: four rows per simdgroup, 16 blocks in flight with two lanes per block, and no nibble shifts (the packed 16-bit words are masked and multiplied by activations pre-divided by 1, 256, 16 and 4096, the -8 offset folded into their sum). In isolation it reads Gemma's matrices at 104-119 GB/s (from 92-94 on the large ones, 69 on q) and Qwen's at 108 (from 90). Gemma's post-norm residual add and the following pre-norm share one kernel pass (two dispatches fewer per layer; no measurable change on its own).

After those, paced:

| | brisk CPU 4 threads | llama.cpp CPU 4 threads | brisk GPU | llama.cpp Metal |
|---|---:|---:|---:|---:|
| prompt | 653-660 | 684 | 1811-1817 | 1924 |
| generation | 118-130 | 117 | 121-134 | 127 |

Level with llama.cpp on three of the four cells and 4-6% behind on prompts. The faster gemv also moved Llama-3.2-1B's GPU generation to 103-104 (llama.cpp 104) and left Qwen3-1.7B's at 88-89.

The stock Q4_K_M download (bartowski; 117 Q5_0 and 14 Q8_0 tensors among the Q4_K/Q6_K ones, including a Q8_0 embedding) runs since Q5_0 and Q8_0 weights were added: perplexity 44.63 against llama.cpp's 44.70 on a 1,650-token text (0.15%; the GPU 1.0% lower, its float activations again), pinned. Its speed: the GPU is close (prompt 1620-1673 vs llama.cpp Metal's 1763, generation 74-83 vs 97). On the CPU the first Q5_0 and Q8_0 kernels were the plain per-row ones (prompt 106-118 tokens/s against llama.cpp's 596 at 4 threads, generation 64-66 vs 73). Two steps since: the plain 4x4 smmla tile that Q4_0 had before its repacked layout now serves Q5_0 and Q8_0 too (prompt 255-277), and Q5_0 got its own four-row interleaved layout, Q5_0x4, repacked at load like Q4_0x4, with the fifth bits stored four bytes per nibble vector so the usmmla tile recovers them with a table lookup and a bit test (prompt 408-482, generation 81). Single-thread, one 6912x1152 matrix at 64 tokens: Q4_0x4 299 GFLOPS, Q5_0x4 218, plain Q5_0 107, plain Q8_0 211; the Q5_0x4 shortfall is the fifth-bit unpack (about 40 vector ops per block against 8 for Q4_0x4). Q8_0 is only the embedding and the attention V in this file, so it keeps the plain tile.

One GPU bug found on the way: Metal's fast-math `tanh` returns NaN for large arguments (its exp overflows), so the GELU epilogue clamps the argument; the CPU's std::tanh had hidden it.

#### Speculative decoding on the GPU

Qwen3-1.7B pure Q4_0, `--gpu`, greedy, lookup (n-gram) drafting, output identical to plain decoding. The Speculator now backs off after a round with no accepted guess (plain single-token rounds, doubling up to 32, halved again on success), which is what keeps the fresh-answer penalty bounded.

| Text | plain | lookup, 5 guesses | 12 guesses | 24 guesses |
|---|---:|---:|---:|---:|
| fresh answer (200 tokens) | 80-90 | 59 (was 42 before the back-off) | | |
| "repeat the text above" (200 tokens) | 78 | 113 | 125 | 234 (99% accepted) |

For comparison the CPU at 4 threads: fresh 68 plain / 54 with lookup; repeat 163 at 12 guesses. The GPU gains more at long drafts (one 25-token verify costs about two single tokens) and less at short ones, because a verify batch of n tokens costs about 0.7 n single tokens on the GPU against about 0.4 n on the CPU: the gemv reads the weights once per activation row. Five kernels were tried to read them once per batch instead (accumulating all rows in one simdgroup; one simdgroup per row sharing a threadgroup; an 8-row simdgroup-matrix kernel in three layouts, with transposed activations and staged slices; paired aligned block loads) and every one was slower than the plain gemv, including at a single token, because on this GPU the extra registers or threadgroup memory cut the resident simdgroups faster than the saved traffic paid back. llama.cpp's `mul_mv_mma` kernel solves the same problem, so a working design exists; it was not reached here.

Measured along the way: a 128x64 tile with 8 simdgroups ran 3x slower (register pressure), as did unrolling the gemv's block loop; halving threadgroup memory per tile changed nothing; block-major 8x8 operand layouts with contiguous `simdgroup_load`s gave +8%; fusing the residual adds and SiLU gate into the matmul epilogues and writing half activations from the producers gave +5%; splitting a single token's attention over position chunks took long-context generation from 68 to 80 tokens/s. A `uint` load at a 2-byte-aligned address inside an 18-byte Q4_0 block is silently rounded down to 4 bytes on this GPU; `metal_test` exists because of it.

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
| Qwen3-1.7B pure Q4_0 | 11.0 | 5.8 | 3.7 | 3.3 |
| BitNet 2B4T TQ2_0 (Q4_0 embedding) | 6.0 | 8.3 | 4.6 | 4.7 |

The 1.7B and BitNet prompt rates are single runs after the batched kernels were blocked by row bands and the K/V cache made head-major (2026-10-14); the 0.6B row predates that. Before those changes the 1.7B prompt rate was 6.6.
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
