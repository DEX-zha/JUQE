"""Build profile for the validated Linux ARM64 / NVIDIA GB10 deployment."""

import os
from pathlib import Path
import platform
import re
import shutil
import subprocess


def executable(name):
    path = shutil.which(name)
    if not path:
        raise ValueError(f"required tool is missing: {name}")
    return str(Path(path).absolute())


def toolchain(root, cuda_compiler):
    if platform.system() != "Linux" or platform.machine() not in ("aarch64", "arm64"):
        raise ValueError("the GB10 build requires Linux ARM64; --prepare-only works on other hosts")
    tools = {name: executable(name) for name in ("git", "cmake", "ninja")}
    tools["cargo"] = executable(os.environ.get("CARGO", "cargo"))
    tools["rustc"] = executable(os.environ.get("RUSTC", "rustc"))
    tools["cc"] = executable(os.environ.get("CC", "cc"))
    tools["cxx"] = executable(os.environ.get("CXX", "c++"))
    tools["nvcc"] = executable(cuda_compiler)
    versions = {name: subprocess.check_output([path, "--version"], text=True, cwd=root).strip()
                for name, path in tools.items()}
    cmake_version = re.search(r"cmake version (\d+)\.(\d+)", versions["cmake"])
    if not cmake_version or tuple(map(int, cmake_version.groups())) < (3, 24):
        raise ValueError("the build requires CMake 3.24 or later")
    if "release 13.0," not in versions["nvcc"]:
        raise ValueError("the pinned GB10 profile requires CUDA Toolkit 13.0")
    return tools, versions


def cmake_command(root, output, rust_library, tools):
    settings = {
        "CMAKE_BUILD_TYPE": "Release", "CMAKE_CUDA_ARCHITECTURES": "121",
        "CMAKE_C_COMPILER": tools["cc"], "CMAKE_CXX_COMPILER": tools["cxx"],
        "CMAKE_CUDA_COMPILER": tools["nvcc"], "BUILD_SHARED_LIBS": "ON",
        # FindCUDAToolkit runs before CUDA language initialization in this engine.
        "CUDAToolkit_ROOT": str(Path(tools["nvcc"]).resolve().parent.parent),
        "GGML_CUDA": "ON", "GGML_NATIVE": "ON", "GGML_CUDA_GRAPHS": "ON",
        "GGML_CUDA_NCCL": "OFF",
        "LLAMA_BUILD_TESTS": "OFF", "LLAMA_BUILD_EXAMPLES": "OFF",
        "LLAMA_BUILD_APP": "OFF", "LLAMA_BUILD_SERVER": "ON",
        "LLAMA_BUILD_UI": "OFF", "LLAMA_USE_PREBUILT_UI": "OFF",
        "JUQE_ROOT": str(root), "JUQE_RUST_LIBRARY": str(rust_library),
    }
    return [tools["cmake"], "--fresh", "-S", str(output / "source"), "-B", str(output / "cmake"),
            "-G", "Ninja", *[f"-D{key}={value}" for key, value in settings.items()]]


def verify_cuda_cache(output, tools):
    root = Path(tools["nvcc"]).resolve().parent.parent
    cache = {}
    for line in (output / "cmake/CMakeCache.txt").read_text().splitlines():
        if line and not line.startswith(("#", "//")) and "=" in line:
            key, value = line.split("=", 1)
            cache[key.split(":", 1)[0]] = value
    for key in ("CUDAToolkit_NVCC_EXECUTABLE", "CUDA_cudart_LIBRARY", "CUDA_cublas_LIBRARY"):
        path = Path(cache.get(key, ""))
        if not path.is_absolute() or not path.exists() or not path.resolve().is_relative_to(root):
            raise ValueError(f"CMake selected {key} outside the chosen CUDA Toolkit: {path}")
