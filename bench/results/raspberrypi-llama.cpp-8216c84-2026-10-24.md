| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen3 0.6B Q4_0                | 319.96 MiB |   596.05 M | CPU        |       4 |           pp512 |         17.99 ± 0.61 |
| qwen3 0.6B Q4_0                | 319.96 MiB |   596.05 M | CPU        |       4 |           tg128 |          9.77 ± 0.17 |
| qwen3 1.7B Q4_0                | 923.39 MiB |     1.72 B | CPU        |       4 |           pp512 |          5.73 ± 0.16 |
| qwen3 1.7B Q4_0                | 923.39 MiB |     1.72 B | CPU        |       4 |           tg128 |          3.40 ± 0.10 |
| bitnet ?B TQ2_0 - 2.06 bpw ternary | 690.21 MiB |     2.41 B | CPU        |       4 |           pp512 |          7.22 ± 0.08 |
| bitnet ?B TQ2_0 - 2.06 bpw ternary | 690.21 MiB |     2.41 B | CPU        |       4 |           tg128 |          4.44 ± 0.15 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | CPU        |       4 |           pp512 |          8.59 ± 0.28 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | CPU        |       4 |           tg128 |          4.25 ± 0.13 |
| llama 3B Q4_K - Medium         |   1.87 GiB |     3.21 B | CPU        |       4 |           pp512 |          2.78 ± 0.03 |
| llama 3B Q4_K - Medium         |   1.87 GiB |     3.21 B | CPU        |       4 |           tg128 |          1.66 ± 0.02 |
| qwen2 1.5B Q4_K - Medium       | 934.69 MiB |     1.54 B | CPU        |       4 |           pp512 |          5.95 ± 0.11 |
| qwen2 1.5B Q4_K - Medium       | 934.69 MiB |     1.54 B | CPU        |       4 |           tg128 |          3.42 ± 0.08 |
| qwen2 1.5B Q4_K - Medium       |   1.04 GiB |     1.78 B | CPU        |       4 |           pp512 |          5.98 ± 0.09 |
| qwen2 1.5B Q4_K - Medium       |   1.04 GiB |     1.78 B | CPU        |       4 |           tg128 |          3.42 ± 0.09 |
| llama ?B Q4_K - Medium         | 1005.01 MiB |     1.71 B | CPU        |       4 |           pp512 |          4.89 ± 0.05 |
| llama ?B Q4_K - Medium         | 1005.01 MiB |     1.71 B | CPU        |       4 |           tg128 |          3.15 ± 0.09 |
| gemma3 1B Q4_0                 | 536.82 MiB |   999.89 M | CPU        |       4 |           pp512 |         11.41 ± 0.38 |
| gemma3 1B Q4_0                 | 536.82 MiB |   999.89 M | CPU        |       4 |           tg128 |          5.86 ± 0.19 |
| gemma3 1B Q4_K - Medium        | 762.49 MiB |   999.89 M | CPU        |       4 |           pp512 |          9.38 ± 0.23 |
| gemma3 1B Q4_K - Medium        | 762.49 MiB |   999.89 M | CPU        |       4 |           tg128 |          4.25 ± 0.11 |
| lfm2 1.2B Q4_0                 | 661.25 MiB |     1.17 B | CPU        |       4 |           pp512 |          8.04 ± 0.20 |
| lfm2 1.2B Q4_0                 | 661.25 MiB |     1.17 B | CPU        |       4 |           tg128 |          4.88 ± 0.16 |
| gemma4 E2B Q4_0                |   2.63 GiB |     4.63 B | CPU        |       4 |           pp512 |          3.70 ± 0.06 |
| gemma4 E2B Q4_0                |   2.63 GiB |     4.63 B | CPU        |       4 |           tg128 |          2.14 ± 0.05 |

build: unknown (0)
