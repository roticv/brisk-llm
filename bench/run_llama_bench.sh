#!/usr/bin/env bash
# Runs llama-bench over the baseline models and writes results to bench/results/.
# Works on both the Mac and the Pi; run it from the repo root.
#
# Env overrides:
#   LLAMA_BIN  path to llama-bench   (default: baselines/llama.cpp/build/bin/llama-bench)
#   LLAMA_COMMIT  llama.cpp commit for the results file name (default: read from its git checkout)
#   MODELS     space-separated GGUFs (default: the Qwen3 4-bit files in models/)
#   THREADS    comma-separated list  (default: perf cores and all cores on macOS, all cores on Linux)
#   NGL        comma-separated list  (default: 0,99 on macOS = CPU and Metal; 0 on Linux)
#   REPS       repetitions per test  (default: 5)
#   DELAY      seconds between tests (default: 20, lets a fanless machine cool down)
set -euo pipefail

LLAMA_BIN=${LLAMA_BIN:-baselines/llama.cpp/build/bin/llama-bench}
MODELS=${MODELS:-"models/Qwen3-0.6B-Q4_0.gguf models/Qwen3-1.7B-Q4_0.gguf models/Qwen3-1.7B-Q4_K_M.gguf"}
REPS=${REPS:-5}
DELAY=${DELAY:-20}

if [[ $(uname) == Darwin ]]; then
    perf=$(sysctl -n hw.perflevel0.physicalcpu)
    all=$(sysctl -n hw.physicalcpu)
    THREADS=${THREADS:-$perf,$all}
    NGL=${NGL:-0,99}
else
    THREADS=${THREADS:-$(nproc)}
    NGL=${NGL:-0}
fi

host=$(hostname -s)
commit=${LLAMA_COMMIT:-$(git -C "$(dirname "$LLAMA_BIN")/../.." rev-parse --short HEAD 2>/dev/null || echo unknown)}
out=bench/results/$host-llama.cpp-$commit
mkdir -p bench/results

model_args=()
for m in $MODELS; do
    [[ -f $m ]] || { echo "missing model: $m" >&2; exit 1; }
    model_args+=(-m "$m")
done

# 512-token prompt and 128-token generation are the llama-bench defaults; stated
# here so brisk-llm's own benchmark can match them exactly.
"$LLAMA_BIN" "${model_args[@]}" -p 512 -n 128 -t "$THREADS" -ngl "$NGL" \
    -r "$REPS" --delay "$DELAY" -o jsonl -oe md > "$out.jsonl" 2> >(tee "$out.md" >&2)

echo "wrote $out.jsonl and $out.md" >&2
