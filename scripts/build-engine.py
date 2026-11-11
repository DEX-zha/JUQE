#!/usr/bin/env python3
"""Build the JUQE adapter beside an existing compatible llama.cpp build."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True, help="JUQE-patched llama.cpp source and build directory")
    parser.add_argument("--output", type=Path, help="Output directory (default: build/llama)")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    engine = args.engine.resolve()
    build = engine / "build"
    output = (args.output or root / "build/llama").resolve()
    if output == build or build in output.parents:
        parser.error("output must be separate from the existing engine build")
    output.mkdir(parents=True, exist_ok=True)
    binary = output / "bin"
    binary.mkdir(exist_ok=True)
    original = build / "bin/libllama.so.0.5.0"
    original_digest = digest(original)
    source = root / "adapters/llama"
    flags = {}
    for line in (build / "src/CMakeFiles/llama.dir/flags.make").read_text().splitlines():
        if line.startswith(("CXX_DEFINES =", "CXX_INCLUDES =", "CXX_FLAGS =")):
            key, value = line.split("=", 1)
            flags[key.strip()] = shlex.split(value)
    link = shlex.split((build / "src/CMakeFiles/llama.dir/link.txt").read_text())
    old_object = "CMakeFiles/llama.dir/juqe-offload.cpp.o"
    if link.count(old_object) != 1:
        parser.error("engine build must contain exactly one JUQE offload object")
    subprocess.run([os.environ.get("CARGO", "cargo"), "build", "--release", "--locked", "-p", "juqe-ffi"], cwd=root, check=True)
    target = Path(os.environ.get("CARGO_TARGET_DIR", root / "target"))
    if not target.is_absolute():
        target = root / target
    library = target / "release/libjuqe_ffi.a"
    commands = []
    objects = []
    for file in sorted(source.glob("*.cpp")):
        obj = output / (file.stem + ".o")
        command = [link[0], *flags["CXX_DEFINES"], "-I" + str(source), "-I" + str(root / "include"),
                   *flags["CXX_INCLUDES"], *flags["CXX_FLAGS"], "-std=c++17", "-Werror",
                   "-c", str(file), "-o", str(obj)]
        subprocess.run(command, check=True)
        commands.append(command)
        objects.append(str(obj))
    shared = binary / original.name
    staged_shared = binary / (original.name + ".tmp")
    link[link.index("-o") + 1] = str(staged_shared)
    position = link.index(old_object)
    link[position:position + 1] = objects + [str(library), "-lpthread", "-ldl", "-lm"]
    subprocess.run(link, cwd=build / "src", check=True)
    staged_shared.replace(shared)
    for name in ("libllama.so", "libllama.so.0"):
        path = binary / name
        if path.is_symlink():
            path.unlink()
        elif path.exists():
            raise RuntimeError(f"refusing to replace a non-symlink: {path}")
        path.symlink_to(shared.name)
    if digest(original) != original_digest:
        raise RuntimeError("existing engine library changed during the build")
    manifest = {"engine": str(engine), "original_library_sha256": original_digest,
                "adapter_library_sha256": digest(shared), "rust_library_sha256": digest(library),
                "compile_commands": commands, "link_command": link,
                "sources_sha256": {str(path.relative_to(root)): digest(path) for path in sorted(source.glob("*")) if path.is_file()}}
    manifest_path = output / "build.json.tmp"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest_path.replace(output / "build.json")
    print(f"JUQE engine library: {shared}")
    print("Existing engine build preserved.")


if __name__ == "__main__":
    main()
