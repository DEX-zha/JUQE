# Pinned engine validation

Date: November 11, 2026, Europe/Paris.

## Tested configuration

- Ubuntu 24.04.4, Linux ARM64, NVIDIA GB10.
- GCC 13.3.0, CMake 3.28.3, Ninja 1.11.1, CUDA Toolkit 13.0.88, Rust 1.99.1.
- Engine commit and patches from [the lock file](../engine/lock.json).
- Existing Qwen3.8-Flash-Next UD-IQ4_XS model and matching expert pack.
- Cache fraction 0.5, four I/O workers, microbatch eight, context 4,096.
- Two HTTP server slots with startup preload enabled.

The engine build directory was initially empty. All engine objects came from the
pinned public source and versioned patches. The build did not consume objects or
libraries from the previous engine installation. The previous driver and loader
were used only as the validation reference, not as build dependencies.

## Results

| Check | Result |
| --- | --- |
| Public source fetch and patched-tree verification | Passed on macOS and Linux |
| Python build and launcher tests | Nine passed on each platform |
| Rust formatting, Clippy, workspace tests, and API docs | Passed on macOS; Rust implementation unchanged |
| Complete CUDA server compilation | Passed with adapter warnings treated as errors |
| Post-build source and artifact verification | Passed |
| Incremental rebuild with fresh CMake configuration | Passed; server and shared-library hashes unchanged |
| Qwen generation comparison | 64 matching token IDs, logit hashes, and maximum logits |
| Concurrent HTTP requests | Expected answers: Paris and 4 |
| Loaded llama/ggml libraries | All resolved inside the new build directory |
| Verification while serving | Passed with the shared build lock |
| Build attempt while serving | Rejected by the exclusive build lock |
| Server shutdown | Clean |
| Previous Qwen service restoration | Healthy |

These are functional installation checks for the tested profile. They do not
establish sustained reliability, broader model support, throughput improvements,
or a 10 ms loading guarantee. Reproducibility means fixed sources and build
instructions; byte-identical output across different host toolchains is not claimed.

## CUDA discovery correction

The first inference attempt exposed mixed Toolkit discovery: CMake used the CUDA
13 compiler but selected CUDA 12 runtime and cuBLAS libraries from another system
installation. Compilation succeeded, but inference failed in `SOFT_MAX` with an
invalid-argument CUDA error.

The build now sets `CUDAToolkit_ROOT` from the selected compiler, starts CMake with
a fresh cache, and checks the selected compiler, runtime, and cuBLAS paths before
compiling. A regression test rejects the mixed configuration. The CUDA objects were
rebuilt with the coherent Toolkit; the successful generation and HTTP checks above
use that corrected build. Dynamic linkage resolves CUDA runtime and cuBLAS version 13.

## Review

Two CodeRabbit reviews completed with no blocking findings at the major threshold.
Corrections cover shared verification locks, avoiding directory creation during
verification, locking before artifact checks, explicit lock-error assertions, the
source map, and final validation status.

The suggestion to disable adapter warnings-as-errors by default was not adopted.
This flag is scoped to JUQE-owned adapter sources and preserves the project's
existing validation policy. It does not promote all upstream or CUDA SDK warnings
to errors. The tested GCC version is recorded above.

The CUDA Toolkit selection fix and its regression test followed the reviews and
were validated through the corrected build and inference run. Final evidence and
status documentation were added after the review snapshots.

See the [machine-readable evidence](evidence/pinned-engine-validation.json) for
source and artifact hashes. Raw model traces and machine-specific build commands
remain outside the repository.
