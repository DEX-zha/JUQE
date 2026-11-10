# JUQE development rules

Read CONTRIBUTING.md and docs/architecture.md before changing runtime behavior.

- This repository is the Rust foundation. Historical C++/Python experiments live
  outside it; never bulk-import model weights, credentials, or raw traces.
- Keep juqe-core free of unsafe code and platform dependencies. Restrict unsafe
  Rust to the audited allocation/syscall boundary in juqe-io with SAFETY comments.
- Preserve exact weights by default. Pruning, quantization changes and prediction
  require separately named experiments and quality evidence.
- Never publish partial reads, evict loading slots or reuse slots held by consumers.
  A GPU consumer must retain its lease until its completion fence, not launch time.
- Measure request-to-ready from submission, including queue wait. Distinguish I/O
  microbenchmarks, inference, resident memory and complete process memory.
- Write all repository content in American English, including documentation,
  comments, CLI messages, review summaries, and commit messages. Follow the
  documentation style in CONTRIBUTING.md.
- Use conventional comments: // explanations, /// public Rust docs, //! module docs,
  // SAFETY: for unsafe invariants. C++ additions use // and Doxygen API comments.
  Explain constraints rather than narrating edits. Keep project reports in docs/.
- Run cargo fmt --all -- --check, cargo clippy --workspace --all-targets -- -D warnings,
  cargo test --workspace --locked, and RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps.
- Validate Linux-specific changes on Linux. macOS tests do not exercise O_DIRECT.
- Use CodeRabbit with .coderabbit.yaml and CONTRIBUTING.md; examine findings,
  fix valid defects and rerun affected checks. Never claim review success from
  configuration alone. Do not enable usage-based billing without user authorization.
- Work on a topic branch. Keep benchmark results and review status accurate.
