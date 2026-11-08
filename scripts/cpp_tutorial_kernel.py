"""Prepare a build-local Jupyter kernel for the litepgo C++ tutorials."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xeus_prefix", type=Path, help="Environment containing xeus-cpp")
    parser.add_argument("--build-dir", type=Path, default=Path("build/release"))
    parser.add_argument("--config", default="Release")
    args = parser.parse_args()

    prefix = args.xeus_prefix.resolve()
    upstream = prefix / "share/jupyter/kernels/xcpp20/kernel.json"
    build = args.build_dir.resolve() / "tutorials/cpp"
    arguments = build / args.config / "kernel-arguments.txt"
    if not upstream.is_file():
        parser.error(f"xeus-cpp C++20 kernel not found at {upstream}")
    if not arguments.is_file():
        parser.error(f"{arguments} is missing. Configure litepgo with CMake first.")
    library, *flags = arguments.read_text().splitlines()
    if not Path(library).is_file():
        parser.error(f"{library} is missing. Build the litepgo_tutorial_cpp target first.")

    spec = json.loads(upstream.read_text())
    spec["display_name"] = "C++20 (litepgo)"
    spec["argv"][0] = str(prefix / "bin/xcpp")
    spec["argv"].extend(flags)
    env = spec.setdefault("env", {})
    if sys.platform == "darwin":
        sdk = subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()
        # Load in Clang so system subprocesses do not inherit a dylib preload.
        spec["argv"].extend(["-isysroot", sdk, "-Xclang", "-load", "-Xclang", library])
    elif sys.platform == "linux":
        # xeus-cpp invokes clang to locate its builtin headers.
        env["PATH"] = f"{prefix / 'bin'}:${{PATH}}"
        env["LD_PRELOAD"] = library
    else:
        parser.error("The litepgo C++ kernel requires macOS or Linux.")

    kernel = build / "jupyter/kernels/litepgo-cpp20"
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel / "kernel.json").write_text(json.dumps(spec, indent=2) + "\n")
    print(f"JUPYTER_PATH={build / 'jupyter'}")


if __name__ == "__main__":
    main()
