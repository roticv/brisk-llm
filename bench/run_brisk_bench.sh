#!/usr/bin/env bash
# Benchmarks brisk the way bench/run_llama_bench.sh benchmarks llama.cpp:
# prompt processing of a ~500-token prompt, and generation of 128 tokens from
# an almost empty context, each repeated with the median reported.
# Run from the repo root.
#
# Env overrides:
#   BRISK      path to brisk               (default: build-release/brisk)
#   MODELS     space-separated GGUFs        (default: the pure Q4_0 files in models/)
#   THREADS    space-separated thread counts (default: perf cores and all cores on macOS, all cores on Linux)
#   REPS       repetitions per test         (default: 5)
set -euo pipefail

BRISK=${BRISK:-build-release/brisk}
MODELS=${MODELS:-"models/Qwen3-0.6B-pure-Q4_0.gguf models/Qwen3-1.7B-pure-Q4_0.gguf"}
REPS=${REPS:-5}
if [[ $(uname) == Darwin ]]; then
    THREADS=${THREADS:-"$(sysctl -n hw.perflevel0.physicalcpu) $(sysctl -n hw.physicalcpu)"}
else
    THREADS=${THREADS:-$(nproc)}
fi

out=bench/results/$(hostname -s)-brisk-$(git rev-parse --short HEAD 2>/dev/null || echo unknown).md
mkdir -p bench/results
{
    echo "| model | threads | prompt tok/s | generation tok/s |"
    echo "|---|---:|---:|---:|"
} > "$out"

median() { sort -n | awk '{ v[NR] = $1 } END { print (NR % 2) ? v[(NR + 1) / 2] : (v[NR / 2] + v[NR / 2 + 1]) / 2 }'; }

for model in $MODELS; do
    for t in $THREADS; do
        prompt_rates=(); gen_rates=()
        for ((r = 0; r < REPS; r++)); do
            line=$("$BRISK" generate "$model" -f bench/prompt512.txt -n 1 -t "$t" --no-stop 2>&1 >/dev/null | tail -1)
            prompt_rates+=("$(sed -E 's/.*prompt [0-9]+ tokens, ([0-9.]+) tokens.*/\1/' <<< "$line")")
            line=$("$BRISK" generate "$model" -p "Once" -n 128 -t "$t" --no-stop 2>&1 >/dev/null | tail -1)
            gen_rates+=("$(sed -E 's/.*generation [0-9]+ tokens, ([0-9.]+) tokens.*/\1/' <<< "$line")")
        done
        p=$(printf '%s\n' "${prompt_rates[@]}" | median)
        g=$(printf '%s\n' "${gen_rates[@]}" | median)
        echo "| $(basename "$model" .gguf) | $t | $p | $g |" | tee -a "$out"
    done
done
echo "wrote $out" >&2
