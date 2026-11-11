# Development conventions

Rust 1.99.1, edition 2024, MIT. The toolchain and Cargo.lock are checked in.
Use American English for all repository content: documentation, code comments,
API names, CLI messages, review instructions, and commit messages. Use rustfmt and
Clippy for Rust formatting and diagnostics. Format C and C++ with clang-format
19.1.7 and the checked-in Microsoft-based `.clang-format` configuration.

## Documentation style

Follow the [Microsoft Writing Style Guide](https://learn.microsoft.com/en-us/style-guide/top-10-tips-style-voice)
for clear wording and sentence-style headings. Use active voice, short sentences,
consistent terminology, and the serial comma. Keep code identifiers in backticks
and specify a language for fenced code blocks.

Follow [GitHub README guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes):
explain the purpose, current status, setup, and support options. Use relative links
for repository files. Keep detailed design and measurement reports in `docs/`.
These are project conventions informed by the guides, not a claim of certification.

## Safety and API design

- The pure core and runtime forbid unsafe code. The C ABI confines raw pointers
  to juqe-ffi and documents engine ownership and GPU completion requirements. The I/O boundary owns aligned allocations and
  keeps buffers exclusively borrowed across synchronous kernel calls.
- Every unsafe block or impl has a `// SAFETY:` comment identifying ownership,
  lifetime, bounds and aliasing assumptions. Do not turn warnings off globally.
- Validate file counts, arithmetic, extents, alignment and allocation bounds before
  issuing I/O. Propagate errors with context; never treat EOF or a partial read as
  a ready expert. No unwrap/expect in runtime code. Tests may use them for assertions.
- Do not silently change I/O mode, precision, pruning or cache capacity. Report the
  actual path and fallback reason. An index validates bounds, not model identity.
- Prefer a minimal dependency set. Add async runtimes or GPU bindings only for a
  concrete integration, with resource limits and an explicit ownership contract.
- The core cache manages metadata. The runtime owns scheduling, and the adapter
  enforces GPU completion before releasing leases. Preserve these boundaries.

## Module structure

Give each file one cohesive responsibility and a descriptive name. Aim for fewer
than 200 lines per implementation file. Split by responsibility rather than line
count; do not scatter one state transition across unrelated modules. Keep module
roots and executable entry points short. Document the source map in
`docs/architecture.md` and the implementation status in `docs/implementation.md`.

## Comments

Use `//` for a non-obvious implementation constraint, `///` for public Rust APIs,
`//!` for module/crate documentation, and `// SAFETY:` for unsafe reasoning.
For C++ adapters use `//` and Doxygen `///` or `/** ... */` API documentation.
Do not add conversational notes, decorative banners, edit histories, or comments
that simply repeat the code. Put architecture and benchmark reports in `docs/`.

## Required validation

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --locked
RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps
```

For Linux I/O on a filesystem supporting O_DIRECT:

```sh
JUQE_REQUIRE_DIRECT=1 cargo test -p juqe-io --locked
```

Tests should cover observable invariants: coalesced requests, failed loads, live
leases, stale/cross-cache completion handles, wrong metadata, truncated files,
unaligned fallbacks and concurrent positional reads. Do not test only the shape
of the implementation. Large model files and real GPU hardware are not CI inputs.

## Review

CodeRabbit's generated assertive profile is extended in `.coderabbit.yaml` with
these conventions and path-specific checks. Before pushing a change:

```sh
coderabbit review --agent --base origin/main --gate major
```

Stage new files first so the reviewed diff includes them. Review feedback is
advisory evidence: assess each finding, test fixes, and record unresolved defects.
Never automatically apply suggestions or describe an unavailable review as passed.
The GitHub CodeRabbit app must be installed on the repository for automatic PR
reviews; the local CLI login alone does not install that app.

## C and C++ checks

```sh
python3 -m venv .local/cpp-format
.local/cpp-format/bin/pip install clang-format==19.1.7
.local/cpp-format/bin/clang-format --dry-run --Werror adapters/llama/*.cpp adapters/llama/*.h include/juqe.h tests/cabi.c
python3 scripts/check-cabi.py
```

## Performance evidence

The current objective is <=10 ms **per expert, request to ready**, including its
queue delay. Record hardware, file geometry, memory budget, concurrent requests,
cache state, exact I/O mode, sample count, p50/p95/p99 and observed maximum.
A measured maximum is not a future hard real-time guarantee. Rust does not imply
an acceleration. Synthetic payloads do not establish model throughput or quality.
