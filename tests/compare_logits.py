#!/usr/bin/env python3
"""Checks brisk's forward pass against llama.cpp on the same model file.

Usage: python3 tests/compare_logits.py <model.gguf> [brisk-binary]

For each prompt this compares the prompt's token ids, the logits that follow
the prompt, and a greedy continuation. Use a float32 model: with 16-bit
weights llama.cpp rounds activations too, so its logits drift from exact maths.
Needs build/llama_ref (see tests/ref/build.sh).
"""

import array
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
GENERATE = 48
# Largest allowed logit difference. Logits span roughly -20 to 30, and the two
# engines sum in different orders, so small float32 differences are expected.
TOLERANCE = 0.05

PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a small village by the sea, there lived",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "<|im_start|>user\nWhat is 17 * 23? Answer briefly.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
    "Explain why the sky is blue in two sentences.<|im_end|>\n<|im_start|>assistant\n",
    "日本の首都は東京です。フランスの首都は",
    "Q: List the first ten prime numbers.\nA: 2, 3, 5, 7,",
    "The following is a detailed technical description of how a transformer language model computes its "
    "output. First, the input text is split into tokens. Each token is mapped to a vector. Then, for every "
    "layer, the model applies self-attention, which lets each position gather information from earlier "
    "positions, followed by a feed-forward network. After the last layer,",
]


def run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"{command[0]} failed:\n{result.stderr}")
    return result.stdout.split("\n")[:-1]


def read_logits(path):
    values = array.array("f")
    values.frombytes(pathlib.Path(path).read_bytes())
    return values


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    model = sys.argv[1]
    brisk = sys.argv[2] if len(sys.argv) == 3 else str(ROOT / "build-release" / "brisk")
    failures = 0

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        for index, prompt in enumerate(PROMPTS):
            prompt_path = tmp / "prompt.txt"
            prompt_path.write_bytes(prompt.encode("utf-8"))
            theirs = run([str(ROOT / "build" / "llama_ref"), "eval", model, str(prompt_path), str(GENERATE),
                          str(tmp / "ref.bin")])
            ours = run([brisk, "generate", model, "-f", str(prompt_path), "-n", str(GENERATE), "--ids", "--no-stop",
                        "--dump-logits", str(tmp / "ours.bin")])

            ref_logits = read_logits(tmp / "ref.bin")
            our_logits = read_logits(tmp / "ours.bin")
            same_size = len(ref_logits) == len(our_logits)
            max_diff = max(abs(a - b) for a, b in zip(ref_logits, our_logits)) if same_size else float("inf")
            same_top = same_size and max(range(len(ref_logits)), key=ref_logits.__getitem__) == max(
                range(len(our_logits)), key=our_logits.__getitem__)

            ref_generated, our_generated = theirs[1].split(), ours[1].split()
            agree = 0
            while agree < min(len(ref_generated), len(our_generated)) and ref_generated[agree] == our_generated[agree]:
                agree += 1

            ok = (theirs[0] == ours[0] and same_top and max_diff <= TOLERANCE
                  and agree == GENERATE == len(our_generated))
            failures += not ok
            print(f"prompt {index}: {len(theirs[0].split()):3d} tokens | prompt ids {'match' if theirs[0] == ours[0] else 'DIFFER'}"
                  f" | max logit diff {max_diff:.5f} | top token {'matches' if same_top else 'DIFFERS'}"
                  f" | greedy {agree}/{GENERATE} | {'ok' if ok else 'FAIL'}")

    print("all prompts match" if failures == 0 else f"{failures} prompt(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
