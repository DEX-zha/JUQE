"""Verify pinned source preparation with a local Git repository and real patches."""

import contextlib
import difflib
import fcntl
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from juqe_build.config import executable, verify_cuda_cache
from juqe_build.source import digest, git, prepare_source, read_lock, verify_source


SPEC = importlib.util.spec_from_file_location(
    "juqe_build_command", Path(__file__).resolve().parents[1] / "scripts/build-engine.py")
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


class EngineSourceTests(unittest.TestCase):
    def test_clean_preparation_reuse_and_modified_source_rejection(self):
        with tempfile.TemporaryDirectory(prefix="juqe-source-") as directory:
            root = Path(directory).resolve()
            upstream = root / "upstream"
            upstream.mkdir()
            git(upstream, "init", "--quiet")
            (upstream / "source.cpp").write_text("original\n")
            git(upstream, "add", ".")
            git(upstream, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                "commit", "--quiet", "-m", "fixture")
            commit = git(upstream, "rev-parse", "HEAD")
            tree = git(upstream, "rev-parse", "HEAD^{tree}")
            patches = root / "engine/patches"
            patches.mkdir(parents=True)
            patch = patches / "adapter.patch"
            patch.write_text("".join(difflib.unified_diff(
                ["original\n"], ["adapted\n"], fromfile="a/source.cpp", tofile="b/source.cpp")))
            git(upstream, "apply", "--index", str(patch))
            lock = {"schema": 1, "repository": str(upstream), "commit": commit, "tree": tree,
                    "patched_tree": git(upstream, "write-tree"),
                    "patches": [{"path": "engine/patches/adapter.patch", "sha256": digest(patch)}]}
            (root / "engine/lock.json").write_text(json.dumps(lock))
            output = root / "output"
            output.mkdir()
            source = prepare_source(root, output, read_lock(root))
            self.assertEqual((source / "source.cpp").read_text(), "adapted\n")
            self.assertEqual(prepare_source(root, output, lock), source)
            (source / "unexpected.cpp").write_text("untracked\n")
            with self.assertRaisesRegex(ValueError, "local changes"):
                verify_source(source, lock)
            (source / "unexpected.cpp").unlink()
            (source / "source.cpp").write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "local changes"):
                verify_source(source, lock)
            git(source, "add", ".")
            with self.assertRaisesRegex(ValueError, "patches changed"):
                verify_source(source, lock)
            patch.write_text("changed patch\n")
            with self.assertRaisesRegex(ValueError, "does not match lock"):
                read_lock(root)

    def test_verify_only_accepts_a_live_shared_lock(self):
        with tempfile.TemporaryDirectory(prefix="juqe-verify-") as directory:
            output = Path(directory).resolve()
            with (output / ".build.lock").open("a") as guard:
                fcntl.flock(guard, fcntl.LOCK_SH)
                with patch("sys.argv", ["build-engine.py", "--output", str(output), "--verify-only"]):
                    with patch.object(BUILD, "verify_source") as source, patch.object(BUILD, "verify_artifacts"):
                        BUILD.main()
                    self.assertEqual(source.call_args.args[0], output / "source")

    def test_verify_only_does_not_create_missing_output(self):
        with tempfile.TemporaryDirectory(prefix="juqe-verify-") as directory:
            output = Path(directory) / "missing"
            with patch("sys.argv", ["build-engine.py", "--output", str(output), "--verify-only"]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    BUILD.main()
            self.assertFalse(output.exists())

    def test_cuda_cache_rejects_mixed_compiler_and_runtime(self):
        with tempfile.TemporaryDirectory(prefix="juqe-cuda-") as directory:
            root = Path(directory).resolve()
            toolkit = root / "cuda-13.0"
            (toolkit / "bin").mkdir(parents=True)
            (toolkit / "lib").mkdir()
            for name in ("bin/nvcc", "lib/libcudart.so", "lib/libcublas.so"):
                (toolkit / name).touch()
            output = root / "output"
            (output / "cmake").mkdir(parents=True)
            cache = output / "cmake/CMakeCache.txt"
            entries = {
                "CUDAToolkit_NVCC_EXECUTABLE": toolkit / "bin/nvcc",
                "CUDA_cudart_LIBRARY": toolkit / "lib/libcudart.so",
                "CUDA_cublas_LIBRARY": toolkit / "lib/libcublas.so",
            }
            cache.write_text("".join(f"{key}:FILEPATH={value}\n" for key, value in entries.items()))
            tools = {"nvcc": str(toolkit / "bin/nvcc")}
            verify_cuda_cache(output, tools)
            other = root / "cuda-12-libcudart.so"
            other.touch()
            cache.write_text(cache.read_text().replace(str(entries["CUDA_cudart_LIBRARY"]), str(other)))
            with self.assertRaisesRegex(ValueError, "CUDA_cudart_LIBRARY outside"):
                verify_cuda_cache(output, tools)

    def test_executable_keeps_dispatcher_symlink_name(self):
        with tempfile.TemporaryDirectory(prefix="juqe-tool-") as directory:
            root = Path(directory)
            dispatcher = root / "dispatcher"
            dispatcher.write_text("#!/bin/sh\nexit 0\n")
            dispatcher.chmod(0o755)
            proxy = root / "cargo"
            proxy.symlink_to(dispatcher)
            self.assertEqual(Path(executable(str(proxy))).name, "cargo")


if __name__ == "__main__":
    unittest.main()
