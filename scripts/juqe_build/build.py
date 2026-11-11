"""Compile all engine artifacts from the pinned source and current Rust runtime."""

import os
from pathlib import Path
import subprocess

from .config import cmake_command, verify_cuda_cache
from .manifest import write_manifest
from .source import verify_source


def compile_engine(root, output, lock, tools, versions, jobs):
    # An interrupted rebuild must not leave an old manifest authorizing new files.
    (output / "build.json").unlink(missing_ok=True)
    cargo = [tools["cargo"], "build", "--release", "--locked", "-p", "juqe-ffi"]
    subprocess.run(cargo, cwd=root, check=True)
    target = Path(os.environ.get("CARGO_TARGET_DIR", root / "target"))
    if not target.is_absolute():
        target = root / target
    rust_library = (target / "release/libjuqe_ffi.a").resolve()
    configure = cmake_command(root, output, rust_library, tools)
    subprocess.run(configure, check=True)
    verify_cuda_cache(output, tools)
    build = [tools["cmake"], "--build", str(output / "cmake"),
             "--target", "llama-server", "--parallel", str(jobs)]
    subprocess.run(build, check=True)
    verify_source(output / "source", lock)
    write_manifest(root, output, lock, versions, [cargo, configure, build], rust_library)
