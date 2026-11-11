# Pinned engine build

## Contract

`python3 scripts/build-engine.py` builds the complete JUQE server from source.
It fetches a fixed llama.cpp commit, applies checked-in integration patches, builds
`juqe-ffi`, and links the Rust static library through CMake. No precompiled engine
objects, private source checkout, or legacy JUQE loader are required.

The supported profile is Linux ARM64 / NVIDIA GB10 / CUDA Toolkit 13.0. It uses
CMake 3.24 or later, Ninja, a C++17 compiler, Python 3.11+, and Rust 1.99.1. See the
[operating guide](operations.md) for installation and server commands.

Reproducibility here means pinned engine sources, verified patches, a documented
build profile, and recorded tool versions and artifacts. It does not promise
byte-identical binaries across different compiler, driver, or system-library
versions. The generated server still needs the CUDA and host runtime libraries.
Warnings in the JUQE-owned adapter are treated as errors; upstream and CUDA SDK
warnings keep their existing policy. This preserves the project's adapter checks.

## Source identity

The lock file records:

- Public repository: `https://github.com/ggml-org/llama.cpp.git`.
- Commit: `3423f940e874e6a97537f2e9854e457f773ad366`.
- Original tree: `663d58663539e84789adc6cbff89809c0c9a7e73`.
- Ordered patch paths and SHA-256 checksums.
- Expected tree after applying those patches.

This commit is the clean upstream base of the engine used in the earlier JUQE
validation. The loader rewrite itself remains in this repository. llama.cpp is
MIT-licensed; its license is retained in `engine/LICENSE.llama.cpp` and in the
fetched checkout. Patches retain upstream style in their surrounding code.

## Integration patches

| Patch | Responsibility |
| --- | --- |
| `0001-expert-lifecycle.patch` | Model ownership, expert tensor replacement, synchronized callbacks, and CMake adapter binding |
| `0002-pinned-host-weights.patch` | Allow the GB10 backend to consume pinned expert pools |
| `0003-callback-graph-identity.patch` | Preserve graph identity for callbacks that change data only |

The source contains hooks for the Qwen4Exp and BailingMoE3 model implementations.
Only the Qwen deployment has complete inference validation. Hooks alone do not
establish support or performance for another model. Legacy pruning, prediction,
two-wave execution, zero-expert kernels, and profiling patches are not imported.

The callback identity optimization requires stable tensor metadata and addresses.
The scheduler's existing backend synchronization remains in place before callbacks;
this build does not change the lease lifetime contract.

## Output and provenance

```text
build/gb10/
  source/          # Isolated pinned checkout with staged integration patches
  cmake/           # Fresh engine build, including bin/llama-server and shared libraries
  build.json       # Source hashes, engine lock, tool versions, commands, artifact hashes
  .build.lock      # Exclusive during preparation/build, shared while serving
```

The launcher verifies the executable and all engine shared libraries, including
symlink aliases. Added or changed libraries cause launch to fail. CUDA and host
system libraries are supplied by the machine and are outside that artifact list.
The engine directory takes precedence in `LD_LIBRARY_PATH`; inherited entries
remain available for system CUDA libraries.

A failed rebuild removes the old completion manifest before compilation. A running
server retains a shared lock across `exec`, so the build cannot overwrite libraries
it is using. Build a new output directory when preparing an upgrade.

## Reuse and recovery

Running the build again verifies the existing source and uses normal Cargo/Ninja
incremental compilation. CMake configuration starts with a fresh cache so stale
compiler and Toolkit selections cannot survive a rebuild. The compiler, CUDA
runtime, and cuBLAS must all resolve inside the selected CUDA 13.0 Toolkit. The
build checks this before compiling. It refuses modified tracked files, changed staged patches,
or untracked files in the engine checkout. It does not reset or delete them.
If a source lock changes or local edits must be preserved, select a new `--output`.

`--prepare-only` fetches and checks the engine without CUDA compilation. This works
on macOS and Linux and runs in Linux CI. `--verify-only` checks an existing source
and artifact manifest without downloading or compiling. Neither command is an
inference test. Full CUDA compilation and HTTP validation require the GB10 host.

The API-only profile disables prebuilt UI downloads. Cargo uses `--locked`, and
Git fetches the exact commit rather than following a branch. OS packages, the CUDA
Toolkit, and the Rust toolchain are prerequisites, not silently installed by JUQE.

## Validation status

The pinned-source build passed on the GB10 host on November 11, 2026. It completed
Qwen generation with 64 records matching the existing loader, two concurrent HTTP
requests, source verification, and live build-lock checks. Source preparation also
passed on macOS; nine build/launcher tests passed on macOS and Linux.

The [pinned build report](engine-build-validation.md) records the tested profile,
CUDA discovery correction, review decisions, and limits. The
[earlier rewrite report](rewrite-validation.md) remains a historical record of the
original overlay build.
