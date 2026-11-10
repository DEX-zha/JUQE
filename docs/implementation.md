# Implementation plan

## Deliverable

Provide an operational JUQE runtime integrated with the existing llama.cpp engine.
Keep the engine's attention, quantization, and CUDA kernels. Rewrite JUQE's expert
management in Rust with explicit ownership and a small C ABI adapter. The runtime
must load a supported pack, run inference, report actionable errors, and shut down
without reusing buffers that are still in use.

The default path preserves weights and routing. Optional legacy behavior must be
identified explicitly; unsupported settings must not silently change results.
Historical benchmark reports are background evidence, not implementation tasks.

## Commit sequence

1. Refactor the core into cache ownership, errors, layouts, and parsing modules.
2. Separate file loading, platform I/O, and request measurements.
3. Separate CLI argument parsing, commands, and output; document module locations.
4. Implement runtime configuration, bounded scheduling, buffer ownership, and shutdown.
5. Add the engine adapter and operational startup and inference commands.
6. Validate complete inference against the existing engine and publish usage guidance.

Each implementation commit includes the relevant tests and documentation. Run
Linux checks for Linux I/O and review changes with CodeRabbit before publication.
Do not describe the port as complete while its engine integration is missing.

## Module rules

- Give each file one cohesive responsibility and a descriptive name.
- Aim for fewer than 200 lines per implementation file. Split by responsibility,
  not at an arbitrary line count; keep related state transitions together.
- Keep entry points and module roots short and easy to navigate.
- Use explicit state and error types rather than combinations of unrelated flags.
- Keep allocation and system calls behind the I/O boundary. The core stays safe
  and independent of platforms and the engine.
- Document public contracts, ownership, and failure conditions in English.
- Preserve the no-partial-publication and no-live-buffer-reuse invariants.

## Progress

- [x] Core cleanup
- [x] I/O cleanup
- [ ] CLI cleanup and source map
- [ ] Operational runtime
- [ ] Engine integration
- [ ] Complete inference validation and operating guide
