#!/usr/bin/env python3
"""Starts brisk serve on a small model and checks the OpenAI-compatible API:
/v1/models, chat completions (plain and streamed), prefix reuse across turns,
stop strings, UTF-8 across token boundaries, and error responses."""
import json, subprocess, sys, time, urllib.request, urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "Qwen3-0.6B-Q4_0.gguf"
PORT = 8097
URL = f"http://127.0.0.1:{PORT}"

def post(path, body, stream=False):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=120)
    if stream:
        events = []
        for line in r:
            line = line.decode().strip()
            if line.startswith("data: "):
                events.append(line[6:])
        return events
    return json.load(r)

def main():
    if not MODEL.exists():
        print("server_test: model missing, skipped")
        return 0
    proc = subprocess.Popen([str(ROOT / "build-release" / "brisk"), "serve", str(MODEL), "--port", str(PORT), "-t", "4"],
                            stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen(URL + "/health", timeout=1)
                break
            except Exception:
                time.sleep(0.2)
        models = json.load(urllib.request.urlopen(URL + "/v1/models"))
        assert models["data"][0]["id"] == "Qwen3-0.6B-Q4_0", models

        msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "What is the capital of France?"}]
        d = post("/v1/chat/completions", {"messages": msgs, "temperature": 0, "enable_thinking": False, "max_tokens": 30})
        answer = d["choices"][0]["message"]["content"]
        assert "Paris" in answer, answer
        assert d["choices"][0]["finish_reason"] == "stop" and d["usage"]["completion_tokens"] > 0

        # The same conversation continued: the answer must be consistent (the reused prefix changes nothing).
        msgs += [{"role": "assistant", "content": answer}, {"role": "user", "content": "And of Italy?"}]
        d2 = post("/v1/chat/completions", {"messages": msgs, "temperature": 0, "enable_thinking": False, "max_tokens": 30})
        assert "Rome" in d2["choices"][0]["message"]["content"], d2

        # Streaming: the concatenated deltas equal the non-streamed answer at temperature 0.
        events = post("/v1/chat/completions", {"messages": msgs[:2], "temperature": 0, "enable_thinking": False,
                                                 "max_tokens": 30, "stream": True}, stream=True)
        assert events[-1] == "[DONE]", events[-1]
        chunks = [json.loads(e) for e in events[:-1]]
        text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
        assert text == answer, (text, answer)
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"

        # A stop string ends the text before it; max_tokens gives "length".
        d = post("/v1/completions", {"prompt": "1, 2, 3, 4,", "temperature": 0, "max_tokens": 30, "stop": ["8"]})
        assert d["choices"][0]["text"].strip().endswith("7,"), d
        assert d["choices"][0]["finish_reason"] == "stop"
        d = post("/v1/completions", {"prompt": "1, 2, 3, 4,", "temperature": 0, "max_tokens": 3})
        assert d["choices"][0]["finish_reason"] == "length" and d["usage"]["completion_tokens"] == 3

        # Multi-byte output arrives intact when streamed.
        events = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "Write the word 'thank you' in Japanese, nothing else."}],
                                                 "temperature": 0, "enable_thinking": False, "max_tokens": 20, "stream": True}, stream=True)
        text = "".join(json.loads(e)["choices"][0]["delta"].get("content", "") for e in events[:-1])
        assert "ありがとう" in text, text

        # Errors are OpenAI-shaped.
        try:
            post("/v1/chat/completions", {"messages": "no"})
            assert False, "expected 400"
        except urllib.error.HTTPError as e:
            assert e.code == 400 and "error" in json.load(e)
        # SIGTERM during a streamed answer: the stream still ends with a finish chunk
        # and [DONE], and the process exits cleanly within a few seconds.
        import signal, threading
        got = []
        def stream():
            got.extend(post("/v1/chat/completions", {"messages": [{"role": "user", "content": "Count slowly from one to two hundred in words."}],
                                                     "temperature": 0, "enable_thinking": False, "max_tokens": 600, "stream": True}, stream=True))
        t = threading.Thread(target=stream)
        t.start()
        time.sleep(0.7)
        proc.send_signal(signal.SIGTERM)
        code = proc.wait(timeout=10)
        t.join(timeout=10)
        assert code == 0, code
        assert got and got[-1] == "[DONE]", got[-2:]
        assert json.loads(got[-2])["choices"][0]["finish_reason"] == "length", got[-2]
        print("server_test: ok")
        return 0
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait()

sys.exit(main())
