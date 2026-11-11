# Architecture

## Scope

The first milestone provides an exact loading path that can be tested without a
GPU. It does not rewrite attention or MoE kernels. The existing llama.cpp engine
remains the reference for outputs and compute time. Prototype performance results
do not describe the new Rust implementation.

`juqe-core` validates metadata and manages per-layer cache state. `juqe-io` performs
synchronous reads into three exclusively borrowed destinations. The planned
scheduler will own buffers and tickets, submit reads, and publish slots only after
success. GPU consumers must retain their leases until their kernels complete.

## Invariants

1. An expert occupies at most one slot in a given cache.
2. A loading slot is never evicted or exposed as ready.
3. All leases on a ready slot must be released before reuse. Clones share an RAII
   guard; dropping the last lease releases that protection. This does not provide
   automatic GPU synchronization.
4. A ticket publishes only its generation in its originating cache. Repeated
   success or failure notifications and stale tickets are rejected. Dropping a
   ticket without an I/O completion leaves the slot unavailable. No `Drop`
   implementation assumes that the writer has finished.
5. A partial load may have written bytes but must not be published. The scheduler
   must call `complete` after `Ok`, or `fail` after the writer returns.
6. LRU eviction applies only to ready, unleased slots. Saturation returns `Busy`
   so the caller can apply backpressure without allocating or forcing eviction.

## Pack format and files

The reader imports the existing text format: a `layers experts` header followed
by 18 integers per layer: identifier, offset, stride, and three groups of
`type/ne0/ne1/off/bytes`. Dense placeholders have a zero stride. The ggml types
remain opaque codes. This milestone validates byte ranges, not the compatibility
of each quantization format with a kernel.

Limits are 256 MiB per block, 4 MiB per index, 512 layers, and 4,096 experts. Extent
arithmetic is checked within the `i64` offset range.

A file-length check does not authenticate a model. A versioned pack manifest with
provenance and cryptographic checksums is planned before GPU integration. Files
must remain immutable during reads. Truncation is detected, but concurrent
same-length rewrites violate the reader contract.

## I/O boundary

Reads are positional, so workers do not share a file cursor. Linux direct mode
explicitly opens the file with `O_DIRECT` and queries alignment constraints with
`statx`. Scatter reads use the three destinations and private scratch regions for
padding.

Incompatible alignment or an `EINVAL`/`EOPNOTSUPP` rejection triggers a full direct
read into aligned staging memory, followed by copies. Disk errors and EOF propagate
to the caller. Interrupted reads are retried, and positive short reads continue.
Byte counts include partial reads performed before a fallback.

`AlignedBuffer` owns initialized host memory, not CUDA memory. The planned adapter
can supply slices from pinned pools. Ownership, address stability, and completion
fences must be established at the C ABI boundary.

## Measurement and integration

The loader accepts the submission timestamp and reports worker wait, service
time, and request-to-ready latency separately. The CLI allocates buffers before
submission and verifies bytes outside the timer. It has no job queue or GPU work.

A 10 ms expert-loading budget does not imply 10 ms per token or Qwen-equivalent
throughput at the same memory budget. Evaluation must account for per-layer cache
misses, bytes transferred, and compute time. Future comparisons must hold the model,
quantization, context, and memory budget constant.

## Source map

| Responsibility | Location |
| --- | --- |
| Cache transitions and eviction | `crates/juqe-core/src/cache/state.rs` |
| Slot states and lease ownership | `crates/juqe-core/src/cache/slot.rs`, `handles.rs` |
| Actionable cache errors | `crates/juqe-core/src/cache/error.rs` |
| Validated pack types | `crates/juqe-core/src/format/index.rs`, `layout.rs` |
| Text format validation | `crates/juqe-core/src/format/parser.rs` |
| Reader contract and timing | `crates/juqe-io/src/reader.rs` |
| Buffered/direct path selection | `crates/juqe-io/src/source.rs` |
| Linux alignment and scatter syscalls | `crates/juqe-io/src/linux/` |
| Aligned allocation ownership | `crates/juqe-io/src/buffer.rs` |
| CLI arguments | `crates/juqe-cli/src/args.rs` |
| CLI operations | `crates/juqe-cli/src/commands/` |
| CLI output | `crates/juqe-cli/src/output.rs` |

Module roots expose the public API. Cache and format tests are separate integration
tests, so they exercise public contracts without depending on internal file layout.
The [implementation plan](implementation.md) tracks the operational rewrite.
