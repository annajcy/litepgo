# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["jupyter-client==8.10.0"]
# ///
"""Build litepgo and prepare its C++20 notebook kernel."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

from jupyter_client.kernelspec import KernelSpecManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-register", action="store_true", help="Leave the kernel in the build directory")
    args = parser.parse_args()

    if sys.platform not in ("darwin", "linux"):
        raise SystemExit("Use macOS or Linux. On Windows, run the setup inside WSL.")
    for tool in ("micromamba", "cmake", "ninja"):
        if shutil.which(tool) is None:
            raise SystemExit(f"Install {tool} and add it to PATH, then rerun this script.")

    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    prefix = root / "build/tutorial-xcpp"
    if (prefix / "conda-meta/history").is_file():
        operation = "install"
    elif prefix.exists() or prefix.is_symlink():
        raise SystemExit(f"{prefix} exists but is not a conda environment. Move it aside and retry.")
    else:
        operation = "create"
    packages = ["xeus-cpp=0.10.0"]
    if sys.platform == "linux":
        packages.append("clang=21")
    subprocess.run(
        ["micromamba", operation, "--yes", "--prefix", str(prefix),
         "--override-channels", "-c", "conda-forge", *packages],
        check=True,
    )

    build = Path("build/cpp-tutorials")
    # shortcut: use scalar Eigen on Linux until the xeus-cpp JIT can load its SIMD headers.
    scalar_eigen = "ON" if sys.platform == "linux" else "OFF"
    subprocess.run(
        ["cmake", "-S", ".", "-B", str(build), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
         "-DLITEPGO_BUILD_TESTS=OFF", "-DLITEPGO_BUILD_PYTHON=OFF",
         "-DLITEPGO_WITH_MKL=OFF", "-DLITEPGO_LINK_COMPILE_COMMANDS=OFF",
         f"-DLITEPGO_EIGEN_DONT_VECTORIZE={scalar_eigen}"],
        check=True,
    )
    subprocess.run(["cmake", "--build", str(build), "--target", "litepgo_tutorial_cpp"], check=True)
    subprocess.run(
        [sys.executable, "scripts/cpp_tutorial_kernel.py", str(prefix), "--build-dir", str(build)],
        check=True,
    )
    if args.no_register:
        return

    source = build / "tutorials/cpp/jupyter/kernels/litepgo-cpp20"
    installed = KernelSpecManager().install_kernel_spec(str(source), user=True)
    print(f"Registered C++20 (litepgo) at {installed}")
    print("If VS Code is already open, run Developer: Reload Window first.")
    print("In the notebook, choose Select Another Kernel > Jupyter Kernel > C++20 (litepgo).")


if __name__ == "__main__":
    main()
