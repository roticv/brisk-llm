ggml_metal_device_init: tensor API disabled for pre-M5 and pre-A19 devices
ggml_metal_library_init: using embedded metal library
ggml_metal_library_compile_all: compiled 'fa_aux' library in 0.001 sec
ggml_metal_library_compile_all: compiled 'fa_f16' library in 0.001 sec
ggml_metal_library_compile_all: compiled 'fa_f32' library in 0.002 sec
ggml_metal_library_compile_all: compiled 'fa_q4_0' library in 0.006 sec
ggml_metal_library_compile_all: compiled 'fa_q4_1' library in 0.002 sec
ggml_metal_library_compile_all: compiled 'fa_q5_0' library in 0.011 sec
ggml_metal_library_compile_all: compiled 'fa_q5_1' library in 0.002 sec
ggml_metal_library_compile_all: compiled 'fa_q8_0' library in 0.003 sec
ggml_metal_library_compile_all: compiled 'fa_vec_f16' library in 0.012 sec
ggml_metal_library_compile_all: compiled 'fa_vec_f32' library in 0.002 sec
ggml_metal_library_compile_all: compiled 'fa_vec_q4_0' library in 0.011 sec
ggml_metal_library_compile_all: compiled 'fa_vec_q4_1' library in 0.009 sec
ggml_metal_library_compile_all: compiled 'fa_vec_q5_0' library in 0.008 sec
ggml_metal_library_compile_all: compiled 'fa_vec_q5_1' library in 0.007 sec
ggml_metal_library_compile_all: compiled 'fa_vec_q8_0' library in 0.004 sec
ggml_metal_library_compile_all: compiled 'mul_mv' library in 0.005 sec
ggml_metal_library_compile_all: compiled 'mul_mv_mma' library in 0.008 sec
ggml_metal_library_compile_all: compiled 'mul_mm' library in 0.003 sec
ggml_metal_library_compile_all: compiled 'quantize' library in 0.010 sec
ggml_metal_library_compile_all: compiled 'softmax' library in 0.003 sec
ggml_metal_library_compile_all: compiled 'norm' library in 0.003 sec
ggml_metal_library_compile_all: compiled 'unary' library in 0.007 sec
ggml_metal_library_compile_all: compiled 'binbcast' library in 0.004 sec
ggml_metal_library_compile_all: compiled 'reduce' library in 0.004 sec
ggml_metal_library_compile_all: compiled 'tri' library in 0.004 sec
ggml_metal_library_compile_all: compiled 'ssm' library in 0.004 sec
ggml_metal_library_compile_all: compiled 'wkv' library in 0.010 sec
ggml_metal_library_compile_all: compiled 'gated_delta_net' library in 0.010 sec
ggml_metal_library_compile_all: compiled 'solve_tri' library in 0.004 sec
ggml_metal_library_compile_all: compiled 'rope' library in 0.004 sec
ggml_metal_library_compile_all: compiled 'conv' library in 0.004 sec
ggml_metal_library_compile_all: compiled 'upscale' library in 0.004 sec
ggml_metal_library_compile_all: compiled 'argsort' library in 0.004 sec
ggml_metal_library_compile_all: compiled 'pool' library in 0.004 sec
ggml_metal_library_compile_all: compiled 'misc' library in 0.004 sec
ggml_metal_library_compile_all: loaded 35 libraries from embedded data in 0.013 sec (max single = 0.012 sec)
ggml_metal_rsets_init: creating a residency set collection (keep_alive = 180 s)
ggml_metal_device_init: GPU name:   MTL0 (Apple M4)
ggml_metal_device_init: GPU family: MTLGPUFamilyApple9  (1009)
ggml_metal_device_init: GPU family: MTLGPUFamilyMetal4  (5002)
ggml_metal_device_init: simdgroup reduction   = true
ggml_metal_device_init: simdgroup matrix mul. = true
ggml_metal_device_init: has unified memory    = true
ggml_metal_device_init: has bfloat            = true
ggml_metal_device_init: has tensor            = false
ggml_metal_device_init: use residency sets    = true
ggml_metal_device_init: use shared buffers    = true
ggml_metal_device_init: recommendedMaxWorkingSetSize  = 26800.60 MB
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen3 0.6B Q4_0                | 358.78 MiB |   596.05 M | MTL,BLAS   |       4 |           pp512 |        933.67 ± 2.17 |
| qwen3 0.6B Q4_0                | 358.78 MiB |   596.05 M | MTL,BLAS   |       4 |           tg128 |        167.03 ± 1.33 |
| qwen3 0.6B Q4_0                | 358.78 MiB |   596.05 M | MTL,BLAS   |      10 |           pp512 |       310.58 ± 69.45 |
| qwen3 0.6B Q4_0                | 358.78 MiB |   596.05 M | MTL,BLAS   |      10 |           tg128 |       129.43 ± 58.81 |
| qwen3 0.6B Q4_0                | 358.78 MiB |   596.05 M | MTL,BLAS   |       4 |           pp512 |     2728.49 ± 131.16 |
| qwen3 0.6B Q4_0                | 358.78 MiB |   596.05 M | MTL,BLAS   |       4 |           tg128 |        174.80 ± 2.40 |
| qwen3 0.6B Q4_0                | 358.78 MiB |   596.05 M | MTL,BLAS   |      10 |           pp512 |      2692.32 ± 10.77 |
| qwen3 0.6B Q4_0                | 358.78 MiB |   596.05 M | MTL,BLAS   |      10 |           tg128 |        158.91 ± 5.78 |
| qwen3 1.7B Q4_0                | 1002.15 MiB |     1.72 B | MTL,BLAS   |       4 |           pp512 |        353.46 ± 9.17 |
| qwen3 1.7B Q4_0                | 1002.15 MiB |     1.72 B | MTL,BLAS   |       4 |           tg128 |         64.75 ± 0.18 |
| qwen3 1.7B Q4_0                | 1002.15 MiB |     1.72 B | MTL,BLAS   |      10 |           pp512 |       401.38 ± 12.54 |
| qwen3 1.7B Q4_0                | 1002.15 MiB |     1.72 B | MTL,BLAS   |      10 |           tg128 |         55.70 ± 5.19 |
| qwen3 1.7B Q4_0                | 1002.15 MiB |     1.72 B | MTL,BLAS   |       4 |           pp512 |        946.43 ± 8.51 |
| qwen3 1.7B Q4_0                | 1002.15 MiB |     1.72 B | MTL,BLAS   |       4 |           tg128 |         78.27 ± 0.39 |
| qwen3 1.7B Q4_0                | 1002.15 MiB |     1.72 B | MTL,BLAS   |      10 |           pp512 |       957.77 ± 11.53 |
| qwen3 1.7B Q4_0                | 1002.15 MiB |     1.72 B | MTL,BLAS   |      10 |           tg128 |         73.53 ± 1.30 |
| qwen3 1.7B Q4_K - Medium       |   1.03 GiB |     1.72 B | MTL,BLAS   |       4 |           pp512 |        223.51 ± 5.73 |
| qwen3 1.7B Q4_K - Medium       |   1.03 GiB |     1.72 B | MTL,BLAS   |       4 |           tg128 |         58.87 ± 0.44 |
| qwen3 1.7B Q4_K - Medium       |   1.03 GiB |     1.72 B | MTL,BLAS   |      10 |           pp512 |        300.72 ± 9.66 |
| qwen3 1.7B Q4_K - Medium       |   1.03 GiB |     1.72 B | MTL,BLAS   |      10 |           tg128 |         53.77 ± 4.10 |
| qwen3 1.7B Q4_K - Medium       |   1.03 GiB |     1.72 B | MTL,BLAS   |       4 |           pp512 |        893.26 ± 6.86 |
| qwen3 1.7B Q4_K - Medium       |   1.03 GiB |     1.72 B | MTL,BLAS   |       4 |           tg128 |         74.78 ± 0.24 |
| qwen3 1.7B Q4_K - Medium       |   1.03 GiB |     1.72 B | MTL,BLAS   |      10 |           pp512 |        897.25 ± 7.39 |
| qwen3 1.7B Q4_K - Medium       |   1.03 GiB |     1.72 B | MTL,BLAS   |      10 |           tg128 |         65.20 ± 2.41 |

build: 8216c84 (1)
