ggml_metal_device_init: tensor API disabled for pre-M5 and pre-A19 devices
ggml_metal_library_init: using embedded metal library
ggml_metal_library_compile_all: compiled 'fa_aux' library in 0.038 sec
ggml_metal_library_compile_all: compiled 'fa_f16' library in 0.034 sec
ggml_metal_library_compile_all: compiled 'fa_f32' library in 0.026 sec
ggml_metal_library_compile_all: compiled 'fa_q4_0' library in 0.040 sec
ggml_metal_library_compile_all: compiled 'fa_q4_1' library in 0.017 sec
ggml_metal_library_compile_all: compiled 'fa_q5_0' library in 0.032 sec
ggml_metal_library_compile_all: compiled 'fa_q5_1' library in 0.030 sec
ggml_metal_library_compile_all: compiled 'fa_q8_0' library in 0.045 sec
ggml_metal_library_compile_all: compiled 'fa_vec_f16' library in 0.037 sec
ggml_metal_library_compile_all: compiled 'fa_vec_f32' library in 0.016 sec
ggml_metal_library_compile_all: compiled 'fa_vec_q4_0' library in 0.043 sec
ggml_metal_library_compile_all: compiled 'fa_vec_q4_1' library in 0.021 sec
ggml_metal_library_compile_all: compiled 'fa_vec_q5_0' library in 0.048 sec
ggml_metal_library_compile_all: compiled 'fa_vec_q5_1' library in 0.025 sec
ggml_metal_library_compile_all: compiled 'fa_vec_q8_0' library in 0.016 sec
ggml_metal_library_compile_all: compiled 'mul_mv' library in 0.012 sec
ggml_metal_library_compile_all: compiled 'mul_mv_mma' library in 0.051 sec
ggml_metal_library_compile_all: compiled 'mul_mm' library in 0.029 sec
ggml_metal_library_compile_all: compiled 'quantize' library in 0.050 sec
ggml_metal_library_compile_all: compiled 'softmax' library in 0.008 sec
ggml_metal_library_compile_all: compiled 'norm' library in 0.006 sec
ggml_metal_library_compile_all: compiled 'unary' library in 0.007 sec
ggml_metal_library_compile_all: compiled 'binbcast' library in 0.009 sec
ggml_metal_library_compile_all: compiled 'reduce' library in 0.029 sec
ggml_metal_library_compile_all: compiled 'tri' library in 0.029 sec
ggml_metal_library_compile_all: compiled 'ssm' library in 0.043 sec
ggml_metal_library_compile_all: compiled 'wkv' library in 0.017 sec
ggml_metal_library_compile_all: compiled 'gated_delta_net' library in 0.008 sec
ggml_metal_library_compile_all: compiled 'solve_tri' library in 0.049 sec
ggml_metal_library_compile_all: compiled 'rope' library in 0.051 sec
ggml_metal_library_compile_all: compiled 'conv' library in 0.052 sec
ggml_metal_library_compile_all: compiled 'upscale' library in 0.043 sec
ggml_metal_library_compile_all: compiled 'argsort' library in 0.043 sec
ggml_metal_library_compile_all: compiled 'pool' library in 0.007 sec
ggml_metal_library_compile_all: compiled 'misc' library in 0.008 sec
ggml_metal_library_compile_all: loaded 35 libraries from embedded data in 0.052 sec (max single = 0.052 sec)
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
| qwen3 0.6B Q4_0                | 319.96 MiB |   596.05 M | MTL,BLAS   |       4 |           pp512 |       697.33 ± 36.54 |
| qwen3 0.6B Q4_0                | 319.96 MiB |   596.05 M | MTL,BLAS   |       4 |           tg128 |        188.87 ± 1.59 |
| qwen3 0.6B Q4_0                | 319.96 MiB |   596.05 M | MTL,BLAS   |      10 |           pp512 |      881.88 ± 219.59 |
| qwen3 0.6B Q4_0                | 319.96 MiB |   596.05 M | MTL,BLAS   |      10 |           tg128 |       161.71 ± 15.19 |
| qwen3 0.6B Q4_0                | 319.96 MiB |   596.05 M | MTL,BLAS   |       4 |           pp512 |      2845.46 ± 26.38 |
| qwen3 0.6B Q4_0                | 319.96 MiB |   596.05 M | MTL,BLAS   |       4 |           tg128 |       197.80 ± 13.10 |
| qwen3 0.6B Q4_0                | 319.96 MiB |   596.05 M | MTL,BLAS   |      10 |           pp512 |      2758.69 ± 67.31 |
| qwen3 0.6B Q4_0                | 319.96 MiB |   596.05 M | MTL,BLAS   |      10 |           tg128 |       165.46 ± 20.19 |
| qwen3 1.7B Q4_0                | 923.39 MiB |     1.72 B | MTL,BLAS   |       4 |           pp512 |        347.06 ± 7.59 |
| qwen3 1.7B Q4_0                | 923.39 MiB |     1.72 B | MTL,BLAS   |       4 |           tg128 |         71.05 ± 0.13 |
| qwen3 1.7B Q4_0                | 923.39 MiB |     1.72 B | MTL,BLAS   |      10 |           pp512 |        425.50 ± 8.32 |
| qwen3 1.7B Q4_0                | 923.39 MiB |     1.72 B | MTL,BLAS   |      10 |           tg128 |         65.72 ± 9.36 |
| qwen3 1.7B Q4_0                | 923.39 MiB |     1.72 B | MTL,BLAS   |       4 |           pp512 |       1002.59 ± 7.01 |
| qwen3 1.7B Q4_0                | 923.39 MiB |     1.72 B | MTL,BLAS   |       4 |           tg128 |         86.25 ± 1.09 |
| qwen3 1.7B Q4_0                | 923.39 MiB |     1.72 B | MTL,BLAS   |      10 |           pp512 |       1015.24 ± 7.66 |
| qwen3 1.7B Q4_0                | 923.39 MiB |     1.72 B | MTL,BLAS   |      10 |           tg128 |        68.93 ± 11.47 |

build: 8216c84 (1)
