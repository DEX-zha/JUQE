# Run JUQE

JUQE uses Rust for expert caching, loading, and batch ownership. The llama.cpp
engine supplies model loading, attention, quantization kernels, and the HTTP server.

## Requirements

The supported inference profile is Linux ARM64 on NVIDIA GB10 with CUDA Toolkit
13.0 and a compatible NVIDIA driver. The standalone Rust components also have
macOS coverage. GPU integration on other hardware is not validated.

On Ubuntu 24.04, install the host build tools:

```sh
sudo apt-get update
sudo apt-get install build-essential cmake ninja-build git python3 libssl-dev
```

Install CUDA Toolkit 13.0 and [rustup](https://rust-lang.org/tools/install/) if they
are not already available. JUQE's checked-in toolchain selects Rust 1.99.1.
Python 3.11 or later and CMake 3.24 or later are required. Keep sufficient memory available for compilation;
the default limits the build to two compiler jobs.

Supply matching GGUF shards and an existing `experts.idx`/`experts.bin` pack on a
filesystem supporting direct I/O. Model weights and pack creation are separate
from engine compilation.

## Build from a clean checkout

```sh
git clone https://github.com/DEX-zha/JUQE.git
cd JUQE
python3 scripts/build-engine.py --jobs 2
```

The script fetches the commit in `engine/lock.json`, verifies the base tree and
patch checksums, applies the versioned patches, and builds Rust plus the complete
CUDA server. It does not read or reuse an earlier llama.cpp installation.
Network access is needed for the pinned Git source and uncached Cargo dependencies.

The default output is `build/gb10`; use `--output /path/to/build` to select another
empty directory. For a different Toolkit location, pass
`--cuda-compiler /path/to/cuda-13.0/bin/nvcc`. The profile requires CUDA 13.0.

```sh
python3 scripts/build-engine.py --verify-only
```

Verification checks the pinned source tree and recorded server artifacts.
See [Pinned engine build](engine-build.md) for provenance, rebuilds, and recovery.

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

The launcher binds to `127.0.0.1` by default. It checks the server and all engine
shared-library hashes before starting and replaces itself with the server process, so normal signals reach the
server directly. Configure this launcher through its arguments; it rejects
inherited `JUQE_*` settings to avoid mixing old configurations with the new runtime.
Use `--build /path/to/build` when the build used a custom output directory. The
launcher holds a shared build lock until the server exits; rebuilding that directory
is refused while it is serving. Use a separate build directory for upgrades.

This profile serves the HTTP API. It does not download or bundle a separate web UI.

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
