"""Record complete local build artifacts and reject changed files at launch."""

import json

from .source import digest


def artifacts(output):
    binary = output / "cmake/bin"
    paths = {binary / "llama-server", binary / "libllama.so"}
    paths.update(binary.glob("*.so*"))
    checksums = {}
    records = {}
    for path in sorted(paths):
        resolved = path.resolve()
        if not resolved.is_relative_to(binary):
            raise ValueError(f"engine artifact escapes the build directory: {path.name}")
        if resolved not in checksums:
            checksums[resolved] = digest(resolved)
        records[str(path.relative_to(output))] = checksums[resolved]
    return records


def write_manifest(root, output, lock, versions, commands, rust_library):
    paths = [root / "Cargo.lock", root / "Cargo.toml", root / "rust-toolchain.toml"]
    for pattern in ("crates/**/*.rs", "crates/**/Cargo.toml", "adapters/llama/*", "include/*.h", "engine/**/*", "scripts/**/*.py"):
        paths.extend(path for path in root.glob(pattern) if path.is_file())
    report = {
        "schema": 2, "profile": "linux-arm64-gb10-cuda13.0", "engine": lock,
        "tool_versions": versions, "commands": commands,
        "source_sha256": {str(path.relative_to(root)): digest(path) for path in sorted(set(paths))},
        "build_inputs_sha256": {
            "rust_static_library": digest(rust_library),
            "cmake_cache": digest(output / "cmake/CMakeCache.txt"),
            "compile_commands": digest(output / "cmake/compile_commands.json"),
        },
        "artifacts_sha256": artifacts(output),
    }
    temporary = output / "build.json.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output / "build.json")


def verify_artifacts(output):
    manifest = output / "build.json"
    if not manifest.is_file():
        raise ValueError("complete engine build is missing; run scripts/build-engine.py first")
    report = json.loads(manifest.read_text())
    if report.get("schema") != 2:
        raise ValueError("legacy adapter build; rebuild with scripts/build-engine.py")
    recorded = report["artifacts_sha256"]
    if not recorded or "cmake/bin/llama-server" not in recorded:
        raise ValueError("build manifest does not contain the server")
    if artifacts(output) != recorded:
        raise ValueError("engine artifacts changed; rebuild before starting the server")
    return output / "cmake/bin/llama-server"
