#!/usr/bin/env python3
"""Compile and run the C ABI integration test against the Rust static library."""

import os
from pathlib import Path
import shlex
import subprocess
import tempfile


def main():
    root = Path(__file__).resolve().parent.parent
    target = Path(os.environ.get("CARGO_TARGET_DIR", root / "target"))
    if not target.is_absolute():
        target = root / target
    subprocess.run([os.environ.get("CARGO", "cargo"), "build", "--locked", "-p", "juqe-ffi"], cwd=root, check=True)
    with tempfile.TemporaryDirectory(prefix="juqe-cabi-") as directory:
        directory = Path(directory)
        binary = directory / "cabi"
        command = shlex.split(os.environ.get("CC", "cc")) + [
            "-std=c11", "-Wall", "-Wextra", "-Werror", "-I", str(root / "include"),
            str(root / "tests/cabi.c"), str(target / "debug/libjuqe_ffi.a"),
            "-lpthread", "-ldl", "-lm", "-o", str(binary),
        ]
        subprocess.run(command, check=True)
        index = directory / "experts.idx"
        weights = directory / "experts.bin"
        index.write_text("1 4\n0 0 4096 0 128 1 0 512 0 128 1 1024 512 0 128 1 2560 512\n")
        weights.write_bytes(bytes(offset % 251 for offset in range(16384)))
        subprocess.run([str(binary), str(index), str(weights)], check=True)


if __name__ == "__main__":
    main()
