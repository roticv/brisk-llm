#!/usr/bin/env python3
"""Checks brisk's perplexity against llama.cpp's on the same model and texts.

Usage: python3 tests/compare_perplexity.py <model.gguf> [brisk-binary]

This is the authoritative correctness check for quantised models: per-logit
comparisons are noisy because 8-bit activation rounding makes the arithmetic
chaotic, but perplexity averages that noise out, so a kernel error shows up as
a systematically worse score. Float32 models agree to four decimals; quantised
ones to about 1%. Needs build/llama_ref (see tests/ref/build.sh).
"""

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TEXTS = [ROOT / "bench" / "prompt512.txt", ROOT / "ROADMAP.md"]
TOLERANCE = 0.02  # relative


def run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"{command[0]} failed:\n{result.stderr}")
    match = re.search(r"perplexity ([0-9.]+)", result.stdout)
    if not match:
        sys.exit(f"{command[0]} printed no perplexity:\n{result.stdout}")
    return float(match.group(1))


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    model = sys.argv[1]
    brisk = sys.argv[2] if len(sys.argv) == 3 else str(ROOT / "build-release" / "brisk")
    failures = 0
    for text in TEXTS:
        ours = run([brisk, "perplexity", model, "-f", str(text)])
        theirs = run([str(ROOT / "build" / "llama_ref"), "perplexity", model, str(text)])
        gap = abs(ours - theirs) / theirs
        ok = gap <= TOLERANCE
        failures += not ok
        print(f"{text.name}: brisk {ours:.4f} | llama.cpp {theirs:.4f} | gap {gap * 100:.2f}% | {'ok' if ok else 'FAIL'}")
    print("perplexities match" if failures == 0 else f"{failures} text(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
