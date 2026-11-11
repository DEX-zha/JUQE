# JUQE

JUQE loads mixture-of-experts (MoE) weights on demand when a model exceeds the
available memory budget. Rust manages the expert cache, parallel reads, and buffer
ownership. A small C++ adapter connects this runtime to the existing llama.cpp
engine and its CUDA kernels.

**Status: operational exact-routing integration on NVIDIA GB10.**
The runtime has completed Qwen generation and concurrent HTTP requests against the
existing engine. See [validation results](docs/rewrite-validation.md) for the tested
scope. The loading target remains at most 10 ms per expert, including queue wait;
it is not a guarantee for every workload or a per-token latency target.

## Features

- Validated import of existing expert packs with bounded metadata and allocations.
- A per-layer LRU cache that protects loading slots and active consumer leases.
- Bounded parallel loading with worker-private scratch and explicit shutdown.
- Linux direct scatter reads into engine-owned pinned pools, with staging fallback.
- An exact llama.cpp adapter with configurable cache capacity and fixed-router compaction.
- Pack inspection, byte verification, C ABI checks, and server startup tools.

The operational path preserves weights and routing. Pruning, learned prediction,
and the experimental two-wave pipeline are not enabled or silently emulated.

## Run the server

Use Linux ARM64 with NVIDIA GB10, CUDA Toolkit 13.0, and the prerequisites in the
[operating guide](docs/operations.md). The build downloads a pinned llama.cpp commit
and compiles the complete server with JUQE; no previous engine build is required.
Model shards and a matching expert pack must already be available.

```sh
cargo test --workspace --locked
python3 scripts/check-cabi.py
python3 scripts/build-engine.py --jobs 2
python3 scripts/serve.py \
  --model /path/to/model-00001-of-00003.gguf \
  --experts /path/to/experts \
  --port 8080
```

The server binds to `127.0.0.1` by default. The launcher does not replace an existing
service. Model weights and engine build artifacts are not included in this repository.

## Inspect and verify a pack

```sh
cargo run --locked -p juqe-cli -- inspect /path/to/experts.idx
cargo run --release --locked -p juqe-cli -- verify /path/to/experts.idx /path/to/experts.bin 0 0 --direct
```

`verify` compares all three matrices with independent positional reads outside the
loader timer. It does not display or modify weights. Its duration measures one
load, not inference throughput. Buffered verification is available on Unix by
omitting `--direct`; direct I/O requires Linux and a compatible filesystem.

## Project structure

| Component | Responsibility |
| --- | --- |
| `juqe-core` | Validated layouts and cache states; no unsafe code or system access |
| `juqe-io` | Aligned allocations, positional reads, and Linux system calls |
| `juqe-runtime` | Bounded loading, batch leases, worker failures, and shutdown |
| `juqe-ffi` | Checked C ABI and access to engine-owned memory |
| `juqe-cli` | Pack inspection and byte verification |
| `adapters/llama` | Tensor binding and synchronized engine callbacks |
| `scripts` | Build, launch, and operational checks |

## Documentation

- [Operating guide](docs/operations.md)
- [Pinned engine build](docs/engine-build.md)
- [Pinned engine validation](docs/engine-build-validation.md)
- [Architecture and source map](docs/architecture.md)
- [Runtime ownership](docs/runtime.md)
- [Engine integration contract](docs/ffi.md)
- [Rewrite validation](docs/rewrite-validation.md)
- [Implementation progress](docs/implementation.md)
- [Historical prototype evidence](docs/evidence.md)

## Contribute and get help

Read the [contribution guidelines](CONTRIBUTING.md) before opening a pull request.
Use [GitHub issues](https://github.com/DEX-zha/JUQE/issues) to report bugs or ask
project questions. Include the platform, command, and relevant error output.
Keep model weights, credentials, and raw traces out of reports and commits.

JUQE is maintained by [DEX-zha](https://github.com/DEX-zha).

## License

JUQE is licensed under the [MIT License](LICENSE).
