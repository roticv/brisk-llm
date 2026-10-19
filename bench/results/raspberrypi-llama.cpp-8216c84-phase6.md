| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen3 0.6B Q4_0                | 319.96 MiB |   596.05 M | CPU        |       4 |           pp512 |         18.91 ± 0.66 |
| qwen3 0.6B Q4_0                | 319.96 MiB |   596.05 M | CPU        |       4 |           tg128 |          9.84 ± 0.14 |
| qwen3 1.7B Q4_0                | 923.39 MiB |     1.72 B | CPU        |       4 |           pp512 |          5.75 ± 0.14 |
| qwen3 1.7B Q4_0                | 923.39 MiB |     1.72 B | CPU        |       4 |           tg128 |          3.40 ± 0.08 |
| bitnet ?B TQ2_0 - 2.06 bpw ternary | 690.21 MiB |     2.41 B | CPU        |       4 |           pp512 |          7.39 ± 0.12 |
| bitnet ?B TQ2_0 - 2.06 bpw ternary | 690.21 MiB |     2.41 B | CPU        |       4 |           tg128 |          4.45 ± 0.14 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | CPU        |       4 |           pp512 |          8.49 ± 0.23 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | CPU        |       4 |           tg128 |          4.15 ± 0.12 |
| qwen2 1.5B Q4_K - Medium       | 934.69 MiB |     1.54 B | CPU        |       4 |           pp512 |          5.99 ± 0.09 |
| qwen2 1.5B Q4_K - Medium       | 934.69 MiB |     1.54 B | CPU        |       4 |           tg128 |          3.42 ± 0.10 |
| gemma3 1B Q4_0                 | 536.82 MiB |   999.89 M | CPU        |       4 |           pp512 |         11.13 ± 0.19 |
| gemma3 1B Q4_0                 | 536.82 MiB |   999.89 M | CPU        |       4 |           tg128 |          5.78 ± 0.22 |
| gemma3 1B Q4_K - Medium        | 762.49 MiB |   999.89 M | CPU        |       4 |           pp512 |          9.50 ± 0.24 |
| gemma3 1B Q4_K - Medium        | 762.49 MiB |   999.89 M | CPU        |       4 |           tg128 |          4.17 ± 0.11 |
| gemma4 E2B Q4_0                |   2.63 GiB |     4.63 B | CPU        |       4 |           pp512 |          3.73 ± 0.06 |
| gemma4 E2B Q4_0                |   2.63 GiB |     4.63 B | CPU        |       4 |           tg128 |          2.12 ± 0.04 |

build: unknown (0)
