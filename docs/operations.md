# Run JUQE

JUQE uses Rust for expert caching, loading, and batch ownership. The llama.cpp
engine supplies model loading, attention, quantization kernels, and the HTTP server.

## Requirements

- Linux ARM64 on NVIDIA GB10, with a compatible direct-I/O filesystem.
  The standalone Rust components also have macOS coverage; GPU integration is
  currently validated on GB10 only.
- The existing JUQE-patched llama.cpp source tree and its completed CMake Makefiles build.
- Rust 1.99.1, a C++17 compiler, and Python 3.11 or later.
- Matching model GGUF shards and an existing `experts.idx`/`experts.bin` pack.

The adapter currently targets the engine already used by this project. It is not
a drop-in plugin for arbitrary llama.cpp or Ollama versions. The build script
requires a separate JUQE offload object in that engine's build.

## Build

From the repository root:

```sh
cargo test --workspace --locked
python3 scripts/check-cabi.py
python3 scripts/build-engine.py --engine /path/to/llama.cpp-juqe
```

The adapter is written to `build/llama`. It reuses the engine's compiled objects
and replaces only its expert-offload object. The original engine build remains
unchanged. `build/llama/build.json` records the libraries and build inputs.

## Start the server

```sh
python3 scripts/serve.py \
  --model /path/to/model-00001-of-00003.gguf \
  --experts /path/to/experts \
  --port 8080 \
  --context 4096 \
  --cache-fraction 0.5 \
  --workers 4
```

The launcher binds to `127.0.0.1` by default. It checks library hashes before
starting and replaces itself with the server process, so normal signals reach the
server directly. Configure this launcher through its arguments; it rejects
inherited `JUQE_*` settings to avoid mixing old configurations with the new runtime.

The default microbatch is eight tokens and the default server slot count is one.
Increase them only within the expert-cache capacity. A microbatch that selects
more distinct experts than available slots cannot run without changing its
execution plan. Increase `--cache-fraction` or reduce `--microbatch` if this occurs.

Use `--preload` to fill each layer's slots before serving requests. A cache fraction
of one also preloads automatically. Verified fixed routers retain compact pools.

## Check health and send a request

```sh
curl --fail http://127.0.0.1:8080/health
curl --fail http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"juqe","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

Run `python3 scripts/check-server.py --url http://127.0.0.1:8080` to check health
and two concurrent chat completion requests.

Stop the server with `Ctrl+C` or `SIGTERM`. The engine must finish GPU work before
closing the runtime. Runtime destruction joins all I/O workers before releasing
pinned pools.

## Supported behavior

The operational path preserves routed experts and their weights. It supports
bounded parallel loading, direct scatter reads with staging fallback, a configurable
cache fraction, fixed-router compaction, preload, and ordinary llama-server APIs.
CPU-affinity overrides and the optional zero-expert kernel naming mode are not
implemented by this adapter and are rejected when enabled. It does not turn on pruning, speculative prediction, or the experimental two-wave
pipeline. Nonzero skip thresholds, test masks, and pipeline settings are rejected.
Legacy research observer installation is also rejected.

A pack's bounds and tensor geometry are checked at startup. These checks do not
prove weight provenance: use the matching verified pack for the model, and keep
its files immutable while the server is running.

## Errors and recovery

Invalid configuration and incompatible packs fail at model startup. Direct I/O
opening failures are reported rather than silently switching to buffered mode.
The runtime never publishes partial reads.

The current engine callback API does not propagate expert-I/O failures as a request
error. A failure during inference therefore stops the process with a diagnostic,
matching the existing engine's fail-stop behavior. Restart only after correcting
the storage or capacity problem. Do not retry inference on incomplete weights.
