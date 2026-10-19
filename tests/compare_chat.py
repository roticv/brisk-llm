#!/usr/bin/env python3
"""Compares brisk's chat rendering with llama.cpp's built-in template renderer
(build/llama_ref template; see tests/ref/build.sh) on a few conversations, for
every model in models/ whose template llama.cpp supports (Gemma 4's and
BitNet's are not in its list, so those are checked by eye against the files'
chat_template)."""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONVERSATIONS = [
    ["user", "Hi there"],
    ["system", "Be brief.", "user", "Hi there"],
    ["system", "Be brief.", "user", "Hi there", "assistant", "Hello! How can I help?", "user", "What is 2+2?"],
    ["user", "  spaces and\nnewlines  ", "assistant", "ok", "user", "again"],
]

def render(prog, model, conv):
    r = subprocess.run([str(prog), "template", str(model)] + conv, capture_output=True)
    if r.returncode != 0:
        return None
    return r.stdout.decode()

def main():
    ok = True
    models = sorted(p for p in (ROOT / "models").glob("*.gguf") if "draft" not in p.name)
    for model in models:
        skipped = 0
        for conv in CONVERSATIONS:
            theirs = render(ROOT / "build" / "llama_ref", model, conv)
            if theirs is None:
                skipped += 1
                continue
            ours = render(ROOT / "build-release" / "brisk", model, conv)
            if ours != theirs:
                ok = False
                print(f"{model.name}: MISMATCH for {conv}\n  ours:   {ours!r}\n  theirs: {theirs!r}")
        print(f"{model.name}: {'not supported by llama_chat_apply_template' if skipped == len(CONVERSATIONS) else 'ok'}")
    print("chat templates match" if ok else "chat templates DIFFER")
    sys.exit(0 if ok else 1)

main()
