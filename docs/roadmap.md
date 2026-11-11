# Runtime roadmap

## Delivered

The operational exact-routing path now includes validated pack import, cache
ownership, bounded loading, direct scatter reads, a checked C ABI, pinned engine
pools, and an HTTP server launcher. The CLI can inspect and verify existing packs. A pinned source build now compiles
the compatible engine directly; no previous engine installation is required.
See the [implementation record](implementation.md) and [validation report](rewrite-validation.md).

## Remaining product work

- Add a versioned pack manifest that authenticates model and weight identity.
  Current validation checks bounds and geometry; operators supply matching packs.
- Extend service recovery beyond the engine's current fail-stop callback contract.
- Validate additional models, hardware, cache budgets, and sustained workloads
  before expanding the supported configuration.
- Measure request-to-ready latency under GPU load before claiming the 10 ms target.
  Report queue delay and tail latency alongside total inference throughput.

Pruning, learned prediction, and the two-wave pipeline remain outside the
operational path. Their historical results do not establish benefits for this
runtime. No Dots3 throughput claim follows from the Qwen integration checks.
