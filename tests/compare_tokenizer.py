#!/usr/bin/env python3
"""Checks that brisk tokenizes exactly like llama.cpp on a corpus of test strings.

Usage: python3 tests/compare_tokenizer.py <model.gguf>

Needs build/brisk and build/llama_ref (see tests/ref/build.sh).
"""

import pathlib
import random
import struct
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent

HANDWRITTEN = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "I'm sure they'll say it's fine, but we've seen that he'd rather not. DON'T! I'M, YOU'LL, 'tis",
    "'s 't 're 've 'm 'll 'd 'x '",
    " leading space",
    "trailing space ",
    "  two  spaces  between   three",
    "tabs\tand\t\ttabs \t mixed",
    "line one\nline two\r\nline three\n\n\nafter blank lines   \n  indented",
    "   \n   ",
    " ",
    "\n",
    "a",
    "3.14159 and 1,000,000 and 0x1F and 1e-9 and 12345678901234567890",
    "price: $42.50 (approx.) -- 50% off!!! #deal @user http://example.com/a?b=c&d=e",
    "def f(x):\n    return x ** 2  # square\n\nfor i in range(10):\n\tprint(f(i))",
    'int main() { std::vector<int> v{1, 2, 3}; return v.size() != 3 ? -1 : 0; } // "ok"',
    "naïve café résumé Zürich Ærøskøbing Łódź",
    "你好,世界!今天天气怎么样?我们去公园散步吧。",
    "こんにちは世界。カタカナとひらがなと漢字。",
    "안녕하세요 세계",
    "Привет, мир! Как дела?",
    "مرحبا بالعالم ١٢٣",
    "שלום עולם",
    "हिन्दी में नमस्ते दुनिया १२३",
    "emoji 😀🎉👍🏽 family 👨‍👩‍👧‍👦 flags 🇸🇬🇯🇵 done",
    "math: ∑ x² ≤ ∞, α + β = γ, ½ ⅓ ²³ Ⅷ ①②③",
    "no-break space, em space, ideographic　space, line sep, nel\u0085here",
    "zero​width and combining é ä and soft­hyphen",
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n4<|im_end|>",
    "<|endoftext|><|im_start|><|im_start|>x<|im_end|",
    "text with <tool_call>{\"name\": \"f\"}</tool_call> and <|fim_prefix|>code<|fim_suffix|>",
    "<|im_start|",
    "",
]

ALPHABET = (
    list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    + list("0123456789") * 2
    + list(" " * 12 + "\n\n\n\r\t\t")
    + list("'''\".,;:!?-_()[]{}<>|/\\@#$%^&*+=~`")
    + list("éüñçøßÆŁ你好世界日本語かなカナ한글Приветمرحباשלוםनमस्ते")
    + list("😀🎉👍½²①∑≤∞  　́​")
)


def random_strings(count):
    rng = random.Random(1234)
    return ["".join(rng.choice(ALPHABET) for _ in range(rng.randint(1, 60))) for _ in range(count)]


def file_chunks():
    chunks = []
    for path in sorted(ROOT.glob("src/*.cpp")) + [ROOT / "ROADMAP.md", ROOT / "bench" / "RESULTS.md"]:
        text = path.read_text(encoding="utf-8")
        chunks.extend(text[i : i + 2000] for i in range(0, len(text), 2000))
    return chunks


def run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"{command[0]} failed:\n{result.stderr}")
    return result.stdout.split("\n")[:-1]


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    model = sys.argv[1]
    corpus = HANDWRITTEN + random_strings(2000) + file_chunks()

    with tempfile.NamedTemporaryFile(suffix=".bin") as f:
        for text in corpus:
            data = text.encode("utf-8")
            f.write(struct.pack("<I", len(data)) + data)
        f.flush()
        ours = run([str(ROOT / "build" / "brisk"), "tokenize", model, "--batch", f.name])
        theirs = run([str(ROOT / "build" / "llama_ref"), "tokenize", model, f.name])

    if len(ours) != len(corpus) or len(theirs) != len(corpus):
        sys.exit(f"expected {len(corpus)} lines, got {len(ours)} from brisk and {len(theirs)} from llama.cpp")

    mismatches = [i for i in range(len(corpus)) if ours[i] != theirs[i]]
    for i in mismatches[:10]:
        print(f"MISMATCH on {corpus[i]!r}\n  brisk:     {ours[i]}\n  llama.cpp: {theirs[i]}")
    tokens = sum(len(line.split()) for line in theirs)
    print(f"{len(corpus) - len(mismatches)}/{len(corpus)} strings match ({tokens} tokens)")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
