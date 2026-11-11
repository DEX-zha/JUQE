#!/usr/bin/env python3
"""Check health and concurrent chat completion requests on a running JUQE server."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import urllib.request


def complete(base_url, prompt):
    payload = {
        "model": "juqe", "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        base_url + "/v1/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    base_url = args.url.rstrip("/")
    with urllib.request.urlopen(base_url + "/health", timeout=10) as response:
        health = json.load(response)
    if health.get("status") != "ok":
        raise RuntimeError(f"server is not ready: {health}")
    prompts = ["Name the capital of France. Answer with one word.",
               "What is 2+2? Answer with one digit."]
    with ThreadPoolExecutor(max_workers=2) as executor:
        replies = list(executor.map(lambda prompt: complete(base_url, prompt), prompts))
    report = {"health": health, "replies": replies}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    for expected, reply in zip(("Paris", "4"), replies):
        content = reply["choices"][0]["message"]["content"]
        if expected not in content:
            raise RuntimeError(f"unexpected server response: {content!r}")
    print("Health and two concurrent chat completion requests passed.")


if __name__ == "__main__":
    main()
