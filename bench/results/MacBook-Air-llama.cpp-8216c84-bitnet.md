ggml_metal_device_init: tensor API disabled for pre-M5 and pre-A19 devices
ggml_metal_library_init: using embedded metal library
ggml_metal_library_compile_all: compiled 'fa_aux' library in 0.002 sec
ggml_metal_library_compile_all: compiled 'fa_f16' library in 0.004 sec
ggml_metal_library_compile_all: compiled 'fa_f32' library in 0.002 sec
ggml_metal_library_compile_all: compiled 'fa_q4_0' library in 0.013 sec
ggml_metal_library_compile_all: compiled 'fa_q4_1' library in 0.005 sec
ggml_metal_library_compile_all: compiled 'fa_q5_0' library in 0.004 sec
ggml_metal_library_compile_all: compiled 'fa_q5_1' library in 0.005 sec
ggml_metal_library_compile_all: compiled 'fa_q8_0' library in 0.012 sec
ggml_metal_library_compile_all: compiled 'fa_vec_f16' library in 0.003 sec
ggml_metal_library_compile_all: compiled 'fa_vec_f32' library in 0.011 sec
ggml_metal_library_compile_all: compiled 'fa_vec_q4_0' library in 0.011 sec
ggml_metal_library_compile_all: compiled 'fa_vec_q4_1' library in 0.010 sec
ggml_metal_library_compile_all: compiled 'fa_vec_q5_0' library in 0.015 sec
ggml_metal_library_compile_all: compiled 'fa_vec_q5_1' library in 0.006 sec
ggml_metal_library_compile_all: compiled 'fa_vec_q8_0' library in 0.007 sec
ggml_metal_library_compile_all: compiled 'mul_mv' library in 0.008 sec
ggml_metal_library_compile_all: compiled 'mul_mv_mma' library in 0.014 sec
ggml_metal_library_compile_all: compiled 'mul_mm' library in 0.013 sec
ggml_metal_library_compile_all: compiled 'quantize' library in 0.012 sec
ggml_metal_library_compile_all: compiled 'softmax' library in 0.011 sec
ggml_metal_library_compile_all: compiled 'norm' library in 0.003 sec
ggml_metal_library_compile_all: compiled 'unary' library in 0.014 sec
ggml_metal_library_compile_all: compiled 'binbcast' library in 0.013 sec
ggml_metal_library_compile_all: compiled 'reduce' library in 0.012 sec
ggml_metal_library_compile_all: compiled 'tri' library in 0.012 sec
ggml_metal_library_compile_all: compiled 'ssm' library in 0.011 sec
ggml_metal_library_compile_all: compiled 'wkv' library in 0.011 sec
ggml_metal_library_compile_all: compiled 'gated_delta_net' library in 0.012 sec
ggml_metal_library_compile_all: compiled 'solve_tri' library in 0.010 sec
ggml_metal_library_compile_all: compiled 'rope' library in 0.012 sec
ggml_metal_library_compile_all: compiled 'conv' library in 0.007 sec
ggml_metal_library_compile_all: compiled 'upscale' library in 0.012 sec
ggml_metal_library_compile_all: compiled 'argsort' library in 0.007 sec
ggml_metal_library_compile_all: compiled 'pool' library in 0.010 sec
ggml_metal_library_compile_all: compiled 'misc' library in 0.010 sec
ggml_metal_library_compile_all: loaded 35 libraries from embedded data in 0.016 sec (max single = 0.015 sec)
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
| bitnet ?B TQ2_0 - 2.06 bpw ternary | 770.94 MiB |     2.41 B | MTL,BLAS   |       4 |           pp512 |        138.85 ± 3.78 |
| bitnet ?B TQ2_0 - 2.06 bpw ternary | 770.94 MiB |     2.41 B | MTL,BLAS   |       4 |           tg128 |         30.04 ± 5.69 |
| bitnet ?B TQ2_0 - 2.06 bpw ternary | 770.94 MiB |     2.41 B | MTL,BLAS   |      10 |           pp512 |        80.83 ± 25.41 |
