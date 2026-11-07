"""Execute Torch or Energy notebooks and export them for VitePress."""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

import nbformat
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient
from nbconvert import MarkdownExporter

ROOT = Path(__file__).resolve().parents[2]
COURSES = ("torch", "energy")
ASSETS = ROOT / "assets"
CPP_KERNELS = ROOT / "build/cpp-tutorials/tutorials/cpp/jupyter/kernels"


def lesson_files(source, metadata):
    for name in metadata.get("tutorial_assets", ()):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+\.veg", name):
            raise ValueError(f"Expected a VEG asset filename, got {name!r}")
        yield ASSETS / name
    for name in metadata.get("tutorial_files", ()):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+\.npy", name):
            raise ValueError(f"Expected a NumPy checkpoint filename, got {name!r}")
        yield source.with_name(name)


def fingerprint(source):
    content = source.read_bytes()
    digest = hashlib.sha256(content)
    for asset in lesson_files(source, json.loads(content)["metadata"]):
        digest.update(asset.read_bytes())
    return digest.hexdigest()


@contextmanager
def tutorial_kernel(notebook):
    if notebook.metadata.kernelspec.name == "litepgo-cpp20":
        if not (CPP_KERNELS / "litepgo-cpp20/kernel.json").is_file():
            raise RuntimeError("Run uv run scripts/setup_cpp_tutorial.py --no-register first.")
        yield AsyncKernelManager(kernel_name="litepgo-cpp20",
                                 kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(CPP_KERNELS)]))
        return

    # A private kernel spec avoids selecting a globally installed Python/kernel.
    with tempfile.TemporaryDirectory() as directory:
        spec = Path(directory) / "litepgo-tutorial"
        spec.mkdir()
        (spec / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "litepgo tutorial", "language": "python",
        }))
        manager = AsyncKernelManager(kernel_name="litepgo-tutorial",
                                kernel_spec_manager=KernelSpecManager(kernel_dirs=[directory]))
        bootstrap = nbformat.v4.new_code_cell(
            f"import sys, pathlib, litepgo\n"
            f"assert sys.prefix == {sys.prefix!r}, 'Wrong tutorial Python environment'\n"
            f"assert pathlib.Path(litepgo.__file__).resolve().is_relative_to(pathlib.Path({str(ROOT)!r})), "
            "'litepgo belongs to another checkout'"
        )
        notebook.cells.insert(0, bootstrap)
        try:
            yield manager
        finally:
            notebook.cells.pop(0)


def execute(source, destination, timeout):
    signature = fingerprint(source)
    notebook = nbformat.read(source, as_version=4)
    notebook.metadata.pop("tutorial_source_sha256", None)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               VECLIB_MAXIMUM_THREADS="1", PYDEVD_DISABLE_FILE_VALIDATION="1")
    # Resolve a relative PYTHONPATH before changing the kernel working directory.
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = os.pathsep.join(str(Path(p).resolve()) for p in env["PYTHONPATH"].split(os.pathsep))
    try:
        with tutorial_kernel(notebook) as manager:
            client = NotebookClient(notebook, km=manager, timeout=timeout, allow_errors=False,
                                    resources={"metadata": {"path": str(source.parent)}})
            client.execute(cleanup_kc=True, env=env)
            notebook.metadata["tutorial_source_sha256"] = signature
    finally:
        destination.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, destination)
    return notebook


def render(source, notebook, pages, downloads, *, course):
    exporter = MarkdownExporter(exclude_input_prompt=True, exclude_output_prompt=True)
    if notebook.metadata.kernelspec.language == "cpp":
        notebook.metadata.language_info.name = "cpp"
    body, resources = exporter.from_notebook_node(notebook, resources={
        "unique_key": source.stem, "output_files_dir": f"assets/{source.stem}",
    })
    pages.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(pages / "assets" / source.stem, ignore_errors=True)
    for name, content in resources.get("outputs", {}).items():
        target = pages / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    downloads.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, downloads / source.name)
    # Vue's base-aware links also work when the site is hosted under /litepgo/.
    header = (f'<!-- Generated from tutorials/{course}; edit the notebook, not this page. -->\n'
              '<script setup>\nimport { withBase } from "vitepress"\n</script>\n\n'
              f'<a :href="withBase(\'/tutorial-notebooks/{course}/{source.name}\')" download>Download notebook</a>')
    if course == "energy":
        lesson, language = source.stem.rsplit("_", 1)
        other, label = ("python", "Python") if language == "cpp" else ("cpp", "C++20")
        header += f' · [{label} version](./{lesson}_{other}.md)'
    for asset in lesson_files(source, notebook.metadata):
        shutil.copyfile(asset, downloads / asset.name)
        header += f' · <a :href="withBase(\'/tutorial-notebooks/{course}/{asset.name}\')" download>{asset.name}</a>'
    header += '\n\n'
    (pages / f"{source.stem}.md").write_text(header + body, encoding="utf-8")


def publish_course(course, args, parser):
    source_dir = ROOT / "tutorials" / course
    cache = ROOT / "build" / f"{course}-tutorials"
    pages = ROOT / "docs/tutorials" / course
    downloads = ROOT / "docs/public/tutorial-notebooks" / course
    available = {path.stem: path for path in sorted(source_dir.glob("[0-9][0-9]_*.ipynb"))}
    if not available or any(name not in available for name in args.lessons):
        parser.error(f"choose lessons from: {', '.join(available)}")
    for name in args.lessons or available:
        source, cached = available[name], cache / f"{name}.ipynb"
        print(f"{'Render' if args.no_execute else 'Execute'} {source.name}", flush=True)
        if args.no_execute:
            notebook = nbformat.read(cached, as_version=4)
            if notebook.metadata.get("tutorial_source_sha256") != fingerprint(source):
                raise ValueError(f"{name}: stale or failed cache; rerun without --no-execute")
        else:
            notebook = execute(source, cached, args.timeout)
        render(source, notebook, pages, downloads, course=course)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lessons", nargs="*", help="Lesson stems within the selected course")
    parser.add_argument("--course", choices=COURSES, help="Select a course; default: both")
    parser.add_argument("--no-execute", action="store_true", help="Render an already executed, source-matching cache")
    parser.add_argument("--timeout", type=int, default=180, help="Timeout per cell in seconds")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    if args.lessons and not args.course:
        parser.error("--course is required with lesson names")
    for course in (args.course,) if args.course else COURSES:
        publish_course(course, args, parser)


if __name__ == "__main__":
    main()
