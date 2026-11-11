#!/usr/bin/env python3
"""Run the compatible llama-server with JUQE's Rust expert runtime."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="First GGUF shard")
    parser.add_argument("--experts", required=True, type=Path, help="Directory containing experts.idx and experts.bin")
    parser.add_argument("--build", type=Path, default=root / "build/llama")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--microbatch", type=int, default=8)
    parser.add_argument("--slots", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache-fraction", type=float, default=0.5)
    parser.add_argument("--preload", action="store_true")
    args = parser.parse_args()
    if not 0 < args.cache_fraction <= 1 or not 1 <= args.workers <= 128:
        parser.error("cache fraction must be in (0, 1]; workers must be in 1..128")
    if not 1 <= args.port <= 65535 or min(args.context, args.microbatch, args.slots) < 1:
        parser.error("port and request capacities must be positive and valid")
    build = args.build.resolve()
    manifest_path = build / "build.json"
    if not manifest_path.is_file():
        parser.error("adapter build is missing; run scripts/build-engine.py first")
    manifest = json.loads(manifest_path.read_text())
    engine = Path(manifest["engine"])
    original = engine / "build/bin/libllama.so.0.5.0"
    library = build / "bin/libllama.so.0.5.0"
    if digest(original) != manifest["original_library_sha256"] or digest(library) != manifest["adapter_library_sha256"]:
        parser.error("engine libraries changed; rebuild the JUQE adapter")
    for path in (args.model, args.experts / "experts.idx", args.experts / "experts.bin"):
        if not path.is_file():
            parser.error(f"required file is missing: {path}")
    inherited = sorted(key for key in os.environ if key.startswith("JUQE_"))
    if inherited:
        parser.error("configure this launcher through arguments; unset inherited settings: " + ", ".join(inherited))
    environment = os.environ.copy()
    environment.update({
        "JUQE_EXPERTS": str(args.experts.resolve()), "JUQE_CACHE_FRAC": str(args.cache_fraction),
        "JUQE_IO_THREADS": str(args.workers), "JUQE_PRELOAD": "1" if args.preload else "0",
        "JUQE_GRAPH_UID": "1", "GGML_CUDA_HOST_WEIGHTS": "1",
        "LD_LIBRARY_PATH": os.pathsep.join(filter(None, [
            str(build / "bin"), str(engine / "build/bin"), os.environ.get("LD_LIBRARY_PATH", "")])),
    })
    server = engine / "build/bin/llama-server"
    command = [str(server), "--model", str(args.model.resolve()), "--host", args.host,
               "--port", str(args.port), "--ctx-size", str(args.context), "--parallel", str(args.slots),
               "--batch-size", str(args.microbatch), "--ubatch-size", str(args.microbatch),
               "--n-gpu-layers", "99", "--load-mode", "none", "--lazy-mode", "on", "--threads", "4"]
    os.execve(server, command, environment)


if __name__ == "__main__":
    main()
