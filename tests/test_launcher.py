"""Launcher regressions that do not require CUDA or model weights."""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("juqe_serve", Path(__file__).resolve().parents[1] / "scripts/serve.py")
SERVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SERVE)


class LauncherTests(unittest.TestCase):
    def test_loader_paths_keep_adapter_first_and_preserve_inherited_entries(self):
        with tempfile.TemporaryDirectory(prefix="juqe-launcher-") as directory:
            root = Path(directory).resolve()
            engine = root / "engine"
            build = root / "adapter"
            for parent in (engine / "build/bin", build / "bin"):
                parent.mkdir(parents=True)
                (parent / "libllama.so.0.5.0").write_bytes(b"fixture")
            digest = hashlib.sha256(b"fixture").hexdigest()
            (build / "build.json").write_text(json.dumps({
                "engine": str(engine), "original_library_sha256": digest,
                "adapter_library_sha256": digest,
            }))
            for name in ("model.gguf", "experts.idx", "experts.bin"):
                (root / name).touch()
            arguments = ["serve.py", "--build", str(build), "--model", str(root / "model.gguf"), "--experts", str(root)]
            for inherited in (None, "", "/opt/cuda/lib64:/opt/driver/lib"):
                with self.subTest(inherited=inherited):
                    environment = {} if inherited is None else {"LD_LIBRARY_PATH": inherited}
                    with patch.dict(os.environ, environment, clear=True), patch("sys.argv", arguments), patch.object(SERVE.os, "execve") as execute:
                        SERVE.main()
                    _, _, passed = execute.call_args.args
                    expected = [str(build / "bin"), str(engine / "build/bin")]
                    if inherited:
                        expected.extend(inherited.split(os.pathsep))
                    self.assertEqual(passed["LD_LIBRARY_PATH"].split(os.pathsep), expected)
                    self.assertEqual(passed["JUQE_EXPERTS"], str(root))

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
