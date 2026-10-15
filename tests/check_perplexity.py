#!/usr/bin/env python3
"""Checks brisk's perplexity against values pinned in tests/perplexity_pins.json.

Usage: python3 tests/check_perplexity.py [--update] [--gpu] [brisk-binary]

Each pin names a model file (under models/) and a text (repo-relative) with
the perplexity brisk produced when the pin was recorded. A kernel or model
regression raises the score well beyond the 1% allowed here; the pins were
themselves verified against llama.cpp with tests/compare_perplexity.py.
Models that are not present are skipped. --update rewrites the pins.
"""

import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PINS = ROOT / "tests" / "perplexity_pins.json"
TOLERANCE = 0.01  # relative


def perplexity(brisk, model, text, gpu):
    """The perplexity, or None when the GPU backend does not take this file."""
    command = [brisk, "perplexity", str(model), "-f", str(text)] + (["--gpu"] if gpu else [])
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        if gpu and "Metal backend needs" in result.stderr:
            return None
        sys.exit(f"brisk perplexity failed on {model.name}:\n{result.stderr}")
    return float(re.search(r"perplexity ([0-9.]+)", result.stdout).group(1))


def main():
    args = [a for a in sys.argv[1:] if a not in ("--update", "--gpu")]
    update = "--update" in sys.argv
    gpu = "--gpu" in sys.argv  # check the Metal backend against the same pins (Q4_0 files only)
    brisk = args[0] if args else str(ROOT / "build-release" / "brisk")
    pins = json.loads(PINS.read_text())

    failures = 0
    checked = 0
    for pin in pins:
        model = ROOT / "models" / pin["model"]
        if not model.exists():
            print(f"{pin['model']}: not present, skipped")
            continue
        value = perplexity(brisk, model, ROOT / pin["text"], gpu)
        if value is None:
            print(f"{pin['model']}: not a pure Q4_0 file, skipped on the GPU")
            continue
        checked += 1
        if update:
            pin["perplexity"] = round(value, 4)
            print(f"{pin['model']} on {pin['text']}: {value:.4f} (pinned)")
            continue
        gap = abs(value - pin["perplexity"]) / pin["perplexity"]
        ok = gap <= TOLERANCE
        failures += not ok
        print(f"{pin['model']} on {pin['text']}: {value:.4f} vs pinned {pin['perplexity']:.4f} | gap {gap * 100:.2f}% | {'ok' if ok else 'FAIL'}")

    if update:
        PINS.write_text(json.dumps(pins, indent=2) + "\n")
        return 0
    if checked == 0:
        sys.exit("no pinned model files are present under models/")
    print("perplexity pins match" if failures == 0 else f"{failures} pin(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
