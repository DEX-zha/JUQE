#!/usr/bin/env python3
"""Fetch the pinned engine, apply verified JUQE patches, and build the GB10 server."""

import argparse
import fcntl
from pathlib import Path
import subprocess

from juqe_build.build import compile_engine
from juqe_build.config import toolchain
from juqe_build.manifest import verify_artifacts
from juqe_build.source import prepare_source, read_lock, verify_source


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "build/gb10")
    parser.add_argument("--jobs", type=int, default=2, help="Parallel compiler jobs (default: 2)")
    parser.add_argument("--cuda-compiler", default="/usr/local/cuda-13.0/bin/nvcc")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true", help="Fetch and verify sources without compiling")
    mode.add_argument("--verify-only", action="store_true", help="Verify an existing source tree and server build")
    args = parser.parse_args()
    if not 1 <= args.jobs <= 128:
        parser.error("jobs must be in 1..128")
    output = args.output.resolve()
    if output == root or output in root.parents:
        parser.error("output must not contain the JUQE source tree")
    try:
        lock = read_lock(root)
        tools, versions = (None, None) if args.prepare_only or args.verify_only else toolchain(root, args.cuda_compiler)
        if args.verify_only:
            if not output.is_dir():
                raise ValueError(f"build directory does not exist: {output}")
        else:
            output.mkdir(parents=True, exist_ok=True)
        with (output / ".build.lock").open("r" if args.verify_only else "a") as guard:
            try:
                lock_mode = fcntl.LOCK_SH if args.verify_only else fcntl.LOCK_EX
                fcntl.flock(guard, lock_mode | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("another build is using this output directory") from error
            if args.verify_only:
                verify_source(output / "source", lock)
                verify_artifacts(output)
                print("Pinned sources and server artifacts verified.")
                return
            prepare_source(root, output, lock)
            if args.prepare_only:
                print(f"Pinned engine sources ready: {output / 'source'}")
                return
            compile_engine(root, output, lock, tools, versions, args.jobs)
            print(f"JUQE server ready: {verify_artifacts(output)}")
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"build failed: {error}\n")


if __name__ == "__main__":
    main()
