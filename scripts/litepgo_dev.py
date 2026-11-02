"""Make a CMake build of litepgo importable from its virtual environment.

Mirrors the Python sources next to the compiled _core in the build tree, then
points a .pth file in the environment's site-packages at that build. Edits to
existing .py files take effect immediately on Unix. Windows copies the sources;
rebuild litepgo_python after editing them. Adding or removing files needs a build.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

PTH_NAME = "litepgo_dev.pth"


def link_sources(source: Path, package: Path) -> None:
    """Mirror Python sources, using copies on Windows to avoid symlink privileges."""
    wanted = {path.relative_to(source) for path in source.rglob("*.py")}
    for relative in wanted:
        link = package / relative
        target = source / relative
        if link.is_symlink() and link.resolve() == target.resolve():
            continue
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            link.unlink()
        if os.name == "nt":
            shutil.copy2(target, link)
        else:
            link.symlink_to(target)
    for link in package.rglob("*.py"):
        if link.relative_to(package) not in wanted:
            link.unlink()


def write_pth(site_packages: Path, source_root: Path, build_python: Path) -> None:
    """Point the .pth at this build, refusing to take over another checkout's registration."""
    registration = site_packages / PTH_NAME
    owner = f"# litepgo checkout: {source_root}\n"
    content = owner + f"{build_python}\n"
    if registration.exists():
        previous = registration.read_text(encoding="utf-8")
        if previous == content:
            return
        if not previous.startswith(owner):
            raise RuntimeError(f"{registration} belongs to another checkout; remove it first.")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=site_packages, delete=False) as temporary:
        temporary.write(content)
    os.replace(temporary.name, registration)


def check_import(package: Path) -> None:
    """Import litepgo in a fresh interpreter and require that it comes from this build."""
    script = "import litepgo, litepgo._core; print(litepgo.__file__); print(litepgo._core.__file__)"
    result = subprocess.run([sys.executable, "-c", script], cwd=sys.prefix, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"importing litepgo failed:\n{result.stderr}")
    for found in result.stdout.splitlines():
        if not Path(found).parent.resolve().is_relative_to(package.resolve()):
            raise RuntimeError(f"another litepgo shadows this build: {found}. Uninstall it or use a fresh venv.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="src/python in the checkout")
    parser.add_argument("--core", type=Path, required=True, help="compiled _core in the build tree")
    arguments = parser.parse_args()
    source = arguments.source.resolve()
    core = arguments.core.resolve()
    package = core.parent
    try:
        if sys.prefix == sys.base_prefix:
            raise RuntimeError(f"{sys.executable} is not a virtual environment; reconfigure CMake with one.")
        if not (source / "__init__.py").is_file() or package.name != "litepgo":
            raise RuntimeError("unexpected source or build layout")
        link_sources(source, package)
        write_pth(Path(sysconfig.get_path("purelib")), source.parents[1], package.parent)
        check_import(package)
    except (OSError, RuntimeError) as error:
        parser.exit(1, f"litepgo_dev: {error}\n")
    print(f"litepgo_dev: {sys.prefix} imports litepgo from {package}")


if __name__ == "__main__":
    main()
