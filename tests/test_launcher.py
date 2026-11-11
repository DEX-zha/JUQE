"""Launcher regressions that do not require CUDA or model weights."""

import contextlib
import fcntl
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from juqe_build.manifest import artifacts, verify_artifacts

SPEC = importlib.util.spec_from_file_location("juqe_serve", SCRIPTS / "serve.py")
SERVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SERVE)


def fixture(root):
    build = root / "build"
    binary = build / "cmake/bin"
    binary.mkdir(parents=True)
    (build / ".build.lock").touch()
    (binary / "llama-server").write_bytes(b"server")
    (binary / "libllama.so.0.5.0").write_bytes(b"library")
    (binary / "libllama.so").symlink_to("libllama.so.0.5.0")
    (build / "build.json").write_text(json.dumps({
        "schema": 2, "artifacts_sha256": artifacts(build),
    }))
    for name in ("model.gguf", "experts.idx", "experts.bin"):
        (root / name).touch()
    return build, ["serve.py", "--build", str(build), "--model", str(root / "model.gguf"),
                   "--experts", str(root)]


class LauncherTests(unittest.TestCase):
    def test_library_paths_and_shared_lock_are_preserved_across_exec(self):
        with tempfile.TemporaryDirectory(prefix="juqe-launcher-") as directory:
            root = Path(directory).resolve()
            build, arguments = fixture(root)

            def check_lock(*_):
                with (build / ".build.lock").open("a") as guard:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)

            for inherited in (None, "", "/opt/cuda/lib64:/opt/driver/lib"):
                with self.subTest(inherited=inherited):
                    environment = {} if inherited is None else {"LD_LIBRARY_PATH": inherited}
                    with patch.dict(os.environ, environment, clear=True), patch("sys.argv", arguments):
                        with patch.object(SERVE.os, "execve", side_effect=check_lock) as execute:
                            SERVE.main()
                    _, _, passed = execute.call_args.args
                    expected = [str(build / "cmake/bin")]
                    if inherited:
                        expected.extend(inherited.split(os.pathsep))
                    self.assertEqual(passed["LD_LIBRARY_PATH"].split(os.pathsep), expected)
                    self.assertEqual(passed["JUQE_EXPERTS"], str(root))

    def test_changed_libraries_aliases_and_unrecorded_libraries_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="juqe-launcher-") as directory:
            build, _ = fixture(Path(directory).resolve())
            binary = build / "cmake/bin"
            verify_artifacts(build)
            (binary / "libllama.so.0.5.0").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "artifacts changed"):
                verify_artifacts(build)
            (binary / "libllama.so.0.5.0").write_bytes(b"library")
            alias = binary / "libllama.so"
            alias.unlink()
            alias.symlink_to("llama-server")
            with self.assertRaisesRegex(ValueError, "artifacts changed"):
                verify_artifacts(build)
            alias.unlink()
            alias.symlink_to("libllama.so.0.5.0")
            (binary / "extra.so").write_bytes(b"unexpected")
            with self.assertRaisesRegex(ValueError, "artifacts changed"):
                verify_artifacts(build)

    def test_active_build_prevents_launch(self):
        with tempfile.TemporaryDirectory(prefix="juqe-launcher-") as directory:
            build, arguments = fixture(Path(directory).resolve())
            with (build / ".build.lock").open("a") as guard:
                fcntl.flock(guard, fcntl.LOCK_EX)
                (build / "build.json").unlink()
                with patch.dict(os.environ, {}, clear=True), patch("sys.argv", arguments):
                    error = io.StringIO()
                    with contextlib.redirect_stderr(error), self.assertRaises(SystemExit) as result:
                        SERVE.main()
                    self.assertEqual(result.exception.code, 2)
                    self.assertIn("build is in progress", error.getvalue())

    def test_missing_build_has_an_actionable_error(self):
        with tempfile.TemporaryDirectory(prefix="juqe-launcher-") as directory:
            error = io.StringIO()
            arguments = ["serve.py", "--build", directory, "--model", "model.gguf", "--experts", "experts"]
            with patch("sys.argv", arguments), contextlib.redirect_stderr(error), self.assertRaises(SystemExit) as result:
                SERVE.main()
            self.assertEqual(result.exception.code, 2)
            self.assertIn("run scripts/build-engine.py first", error.getvalue())


if __name__ == "__main__":
    unittest.main()
