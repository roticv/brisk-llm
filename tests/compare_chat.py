#!/usr/bin/env python3
"""Compares brisk's chat rendering with llama.cpp's Jinja engine (what llama-server
renders; build/llama_ref jinja, see tests/ref/build.sh) on OpenAI-style requests:
plain conversations, thinking on and off, and tool calls, for every model in
models/.

Known, deliberate differences: brisk refuses tools for a template that has no
tool format (llama.cpp renders the conversation without them), and brisk does
not render Gemma 4's, DeepSeek's, Mistral's or Phi's tool formats yet. Gemma 4
is compared with thinking off (brisk does not render its thinking channel)."""
import json, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}},
                   "required": ["city"]}}}]
CALL = {"id": "c0", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}
REQUESTS = {
    "plain": {"messages": [{"role": "user", "content": "Hi there"}]},
    "system": {"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi there"}]},
    "turns": {"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi there"},
                           {"role": "assistant", "content": "Hello! How can I help?"}, {"role": "user", "content": "What is 2+2?"}]},
    "spaces": {"messages": [{"role": "user", "content": "  spaces and\nnewlines  "}, {"role": "assistant", "content": "ok"},
                            {"role": "user", "content": "again"}]},
    "no-think": {"messages": [{"role": "user", "content": "Hi there"}], "enable_thinking": False},
    "tools": {"messages": [{"role": "user", "content": "Weather in Paris?"}], "tools": TOOLS},
    "tools-system": {"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Weather in Paris?"}],
                     "tools": TOOLS},
    "tool-round": {"messages": [{"role": "user", "content": "Weather in Paris?"},
                                {"role": "assistant", "content": "", "tool_calls": [CALL]},
                                {"role": "tool", "tool_call_id": "c0", "content": '{"temperature_c": 14}'}], "tools": TOOLS},
}
# Which of brisk's formats take tools, by a marker in the file's template.
TOOL_TEMPLATES = ("<|im_start|>", "<|start_header_id|>")


def render(prog, mode, model, request):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(request, f)
    r = subprocess.run([str(prog), mode, str(model)] + (["--json", f.name] if mode == "template" else [f.name]),
                       capture_output=True)
    Path(f.name).unlink()
    return r.returncode, r.stdout.decode(errors="replace"), r.stderr.decode(errors="replace")


def main():
    ok = True
    models = sorted(p for p in (ROOT / "models").glob("*.gguf") if "draft" not in p.name)
    for model in models:
        gemma4 = "gemma-4" in model.name
        for name, request in REQUESTS.items():
            request = dict(request)
            if gemma4 and "enable_thinking" not in request:
                request["enable_thinking"] = False
            code, theirs, err = render(ROOT / "build" / "llama_ref", "jinja", model, request)
            if code != 0:
                reason = err.strip().splitlines()[-1] if err.strip() else str(code)
                if "tools" in request and "roles must alternate" in reason:
                    continue  # a template without tools: llama.cpp cannot place the tool turn either
                print(f"{model.name}: llama.cpp cannot render {name} ({reason})")
                break
            ours_code, ours, ours_err = render(ROOT / "build-release" / "brisk", "template", model, request)
            if ours_code != 0:
                if "tools" in request and (gemma4 or "<|im_start|>" not in theirs and "<|start_header_id|>" not in theirs
                                           or "no tool format" in ours_err):
                    continue  # brisk declines tools this template cannot carry
                ok = False
                print(f"{model.name}: brisk failed on {name}: {ours_err.strip()}")
                continue
            if ours != theirs:
                ok = False
                print(f"{model.name}: MISMATCH on {name}\n  ours:   {ours!r}\n  theirs: {theirs!r}")
        else:
            print(f"{model.name}: ok")
    print("chat templates match" if ok else "chat templates DIFFER")
    sys.exit(0 if ok else 1)


main()
