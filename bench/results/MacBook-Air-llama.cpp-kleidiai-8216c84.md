ggml_metal_device_init: tensor API disabled for pre-M5 and pre-A19 devices
ggml_metal_library_init: using embedded metal library
ggml_metal_library_compile_all: compiled 'fa_aux' library in 0.155 sec
ggml_metal_library_compile_all: compiled 'fa_f16' library in 4.941 sec
ggml_metal_library_compile_all: compiled 'fa_f32' library in 2.340 sec
ggml_metal_library_compile_all: compiled 'fa_q4_0' library in 9.149 sec
ggml_metal_library_compile_all: compiled 'fa_q4_1' library in 5.667 sec
ggml_metal_library_compile_all: compiled 'fa_q5_0' library in 0.741 sec
ggml_metal_library_compile_all: compiled 'fa_q5_1' library in 1.303 sec
ggml_metal_library_compile_all: compiled 'fa_q8_0' library in 1.916 sec
ggml_metal_library_compile_all: compiled 'fa_vec_f16' library in 9.034 sec
ggml_metal_library_compile_all: compiled 'fa_vec_f32' library in 5.108 sec
ggml_metal_library_compile_all: compiled 'fa_vec_q4_0' library in 1.783 sec
ggml_metal_library_compile_all: compiled 'fa_vec_q4_1' library in 6.918 sec
ggml_metal_library_compile_all: compiled 'fa_vec_q5_0' library in 4.938 sec
ggml_metal_library_compile_all: compiled 'fa_vec_q5_1' library in 7.242 sec
ggml_metal_library_compile_all: compiled 'fa_vec_q8_0' library in 8.558 sec
ggml_metal_library_compile_all: compiled 'mul_mv' library in 3.237 sec
ggml_metal_library_compile_all: compiled 'mul_mv_mma' library in 2.343 sec
ggml_metal_library_compile_all: compiled 'mul_mm' library in 3.665 sec
ggml_metal_library_compile_all: compiled 'quantize' library in 3.237 sec
ggml_metal_library_compile_all: compiled 'softmax' library in 3.256 sec
ggml_metal_library_compile_all: compiled 'norm' library in 4.936 sec
ggml_metal_library_compile_all: compiled 'unary' library in 2.535 sec
ggml_metal_library_compile_all: compiled 'binbcast' library in 2.526 sec
ggml_metal_library_compile_all: compiled 'reduce' library in 2.494 sec
ggml_metal_library_compile_all: compiled 'tri' library in 2.492 sec
ggml_metal_library_compile_all: compiled 'ssm' library in 2.492 sec
ggml_metal_library_compile_all: compiled 'wkv' library in 2.460 sec
ggml_metal_library_compile_all: compiled 'gated_delta_net' library in 2.464 sec
ggml_metal_library_compile_all: compiled 'solve_tri' library in 2.447 sec
ggml_metal_library_compile_all: compiled 'rope' library in 2.436 sec
ggml_metal_library_compile_all: compiled 'conv' library in 2.444 sec
ggml_metal_library_compile_all: compiled 'upscale' library in 2.404 sec
ggml_metal_library_compile_all: compiled 'argsort' library in 2.404 sec
ggml_metal_library_compile_all: compiled 'pool' library in 2.345 sec
ggml_metal_library_compile_all: compiled 'misc' library in 2.404 sec
ggml_metal_library_compile_all: loaded 35 libraries from embedded data in 9.149 sec (max single = 9.149 sec)
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
| qwen3 1.7B Q4_0                | 1002.15 MiB |     1.72 B | MTL,BLAS   |       4 |           pp512 |       390.48 ± 24.52 |
| qwen3 1.7B Q4_0                | 1002.15 MiB |     1.72 B | MTL,BLAS   |       4 |           tg128 |         63.55 ± 0.78 |
| qwen3 0.6B Q4_0                | 358.78 MiB |   596.05 M | MTL,BLAS   |       4 |           pp512 |      1001.92 ± 42.43 |
| qwen3 0.6B Q4_0                | 358.78 MiB |   596.05 M | MTL,BLAS   |       4 |           tg128 |        165.89 ± 0.97 |

build: 8216c84 (1)
