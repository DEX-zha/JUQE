# JUQE

JUQE provides expert caching and loading for mixture-of-experts (MoE) models that
exceed the available memory budget. This Rust foundation carries forward the
invariants tested in C++ and Python prototypes on NVIDIA GB10.

**Status: working foundation. GPU inference integration is still in development.**
The loading target is at most 10 ms per expert from request to readiness, including
queue wait. This target is not guaranteed for all workloads.

## Features

- Bounded, validated import of the existing `experts.idx` format.
- A per-layer LRU cache that coalesces requests, protects loading and leased slots,
  and publishes completed loads explicitly.
- Positional reads on Unix and `preadv` with `O_DIRECT` on Linux, with an aligned
  staging fallback and reporting of the actual read path.
- A CLI to inspect packs and verify all three matrices of an expert.
- Invariant tests, Linux and macOS CI configuration, and CodeRabbit review rules.

The current Rust buffers use ordinary host memory. The bounded scheduler is implemented; CUDA pinned pools and the llama.cpp adapter
are the next integration step. This version preserves exact
weights; pruning and learned prediction are not enabled.

## Get started

Install [rustup](https://rust-lang.org/tools/install/), then run these commands from
the repository root. Rustup selects the pinned toolchain automatically. The CLI
requires Unix; direct I/O requires Linux and a compatible filesystem.

```sh
cargo test --workspace --locked
cargo run --locked -p juqe-cli -- inspect /path/to/experts.idx
cargo run --release --locked -p juqe-cli -- verify /path/to/experts.idx /path/to/experts.bin 0 0
```

To verify direct I/O on Linux:

```sh
cargo run --release --locked -p juqe-cli -- verify /path/to/experts.idx /path/to/experts.bin 0 0 --direct
JUQE_REQUIRE_DIRECT=1 cargo test -p juqe-io --locked
```

Provide an existing `experts.idx` and `experts.bin` pair. Model weights are not
included in this repository. See the [pack format](docs/architecture.md#pack-format-and-files)
for the import contract.

`verify` compares each matrix with an independent read outside the loader timer.
It does not display or modify weights. The reported duration measures one load,
not model throughput. These commands do not control GPU services.

## Project structure

| Crate | Responsibility |
| --- | --- |
| `juqe-core` | Validated metadata and cache states, without unsafe code or system access |
| `juqe-io` | Aligned buffers and exact reads at the allocation and system-call boundary |
| `juqe-runtime` | Bounded parallel loading, batch leases, and worker shutdown |
| `juqe-cli` | Inspection and verification of existing packs |

## Documentation

- [Architecture and invariants](docs/architecture.md)
- [Prototype evidence](docs/evidence.md)
- [Validation results](docs/validation.md)
- [Runtime ownership](docs/runtime.md)
- [Implementation progress](docs/implementation.md)
- [Migration roadmap](docs/roadmap.md)

## Contribute and get help

Read the [contribution guidelines](CONTRIBUTING.md) before opening a pull request.
Use [GitHub issues](https://github.com/DEX-zha/JUQE/issues) to report bugs or ask
project questions. Include the platform, command, and relevant error output.
Keep model weights, credentials, and raw traces out of reports and commits.

JUQE is maintained by [DEX-zha](https://github.com/DEX-zha).

## License

JUQE is licensed under the [MIT License](LICENSE).
