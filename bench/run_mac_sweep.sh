#!/usr/bin/env bash
# One row per model: brisk and llama.cpp on the CPU (4 threads) and the GPU,
# prompt (~500 tokens, --warmup) and generation (64 tokens), single runs with
# pauses. Writes a markdown table. Run from the repo root on an idle machine.
set -euo pipefail
MODELS=${MODELS:-"Qwen3-0.6B-Q4_0 Qwen3-1.7B-pure-Q4_0 Qwen3-1.7B-Q4_K_M Qwen2.5-1.5B-Instruct-Q4_K_M DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M Llama-3.2-1B-Instruct-Q4_K_M Llama-3.2-3B-Instruct-Q4_K_M Mistral-7B-Instruct-v0.3-Q4_K_M SmolLM2-1.7B-Instruct-Q4_K_M gemma-3-1b-it-pure-Q4_0 gemma-3-1b-it-Q4_K_M gemma-4-E2B-it-Q4_0 gemma-4-E4B-it-Q4_0 Phi-4-mini-instruct-Q4_K_M LFM2-1.2B-Q4_0 bitnet-2B4T-TQ2_0-e4"}
out=bench/results/$(hostname -s)-sweep-$(git rev-parse --short HEAD).md
pause() { perl -e 'select(undef, undef, undef, 8)'; }
rate() { sed -E "s/.*$1 [0-9]+ tokens, ([0-9.]+) tokens.*/\1/"; }
{
    echo "| File | brisk CPU 4t pp / tg | llama.cpp CPU 4t pp / tg | brisk GPU pp / tg | llama.cpp Metal pp / tg |"
    echo "|---|---:|---:|---:|---:|"
} > "$out"
for m in $MODELS; do
    f=models/$m.gguf
    [[ -f $f ]] || continue
    pause; cpp=$(build-release/brisk generate "$f" -f bench/prompt512.txt -n 1 -t 4 --warmup 2>&1 >/dev/null | tail -1 | rate prompt)
    pause; ctg=$(build-release/brisk generate "$f" -p Once -n 64 -t 4 --no-stop 2>&1 >/dev/null | tail -1 | rate generation)
    pause; gpp=$(build-release/brisk generate "$f" -f bench/prompt512.txt -n 1 --gpu --warmup 2>&1 >/dev/null | tail -1 | rate prompt)
    pause; gtg=$(build-release/brisk generate "$f" -p Once -n 64 --gpu --no-stop 2>&1 >/dev/null | tail -1 | rate generation)
    pause
    l=$(baselines/llama.cpp/build/bin/llama-bench -m "$f" -p 512 -n 64 -t 4 -ngl 0,99 -r 2 2>/dev/null | grep -E "pp512|tg64" | awk -F'|' '{gsub(/ /,"",$8); split($8,a,"±"); printf "%s ", a[1]}')
    read -r lcpp lctg lgpp lgtg <<< "$l"
    printf '| %s | %s / %s | %.0f / %.1f | %s / %s | %.0f / %.1f |\n' "$m" "$cpp" "$ctg" "$lcpp" "$lctg" "$gpp" "$gtg" "$lgpp" "$lgtg" | tee -a "$out"
done
echo "wrote $out" >&2
