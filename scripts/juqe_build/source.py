"""Prepare and verify an isolated engine checkout from the checked-in lock."""

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git(source, *arguments):
    return subprocess.check_output(
        ["git", "-C", str(source), *arguments], text=True
    ).strip()


def read_lock(root):
    lock = json.loads((root / "engine/lock.json").read_text())
    if lock["schema"] != 1:
        raise ValueError("unsupported engine lock schema")
    for entry in lock["patches"]:
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root / "engine/patches") or digest(path) != entry["sha256"]:
            raise ValueError(f"engine patch does not match lock: {entry['path']}")
    return lock


def verify_source(source, lock):
    if git(source, "rev-parse", "HEAD") != lock["commit"]:
        raise ValueError("engine checkout has a different commit; use a new build directory")
    if git(source, "rev-parse", "HEAD^{tree}") != lock["tree"]:
        raise ValueError("engine base tree does not match lock")
    if git(source, "write-tree") != lock["patched_tree"]:
        raise ValueError("engine patches changed; use a new build directory")
    if git(source, "diff", "--name-only") or git(source, "ls-files", "--others"):
        raise ValueError("engine source has local changes; preserve them and use a new build directory")


def prepare_source(root, output, lock):
    source = output / "source"
    if source.exists():
        verify_source(source, lock)
        return source
    with tempfile.TemporaryDirectory(prefix="source-", dir=output) as directory:
        staging = Path(directory) / "checkout"
        staging.mkdir()
        git(staging, "init", "--quiet")
        git(staging, "fetch", "--depth=1", "--no-tags", lock["repository"], lock["commit"])
        git(staging, "checkout", "--quiet", "--detach", "FETCH_HEAD")
        if git(staging, "rev-parse", "HEAD^{tree}") != lock["tree"]:
            raise ValueError("downloaded engine tree does not match lock")
        patches = [str(root / entry["path"]) for entry in lock["patches"]]
        git(staging, "apply", "--check", *patches)
        git(staging, "apply", "--index", *patches)
        verify_source(staging, lock)
        staging.rename(source)
    return source
