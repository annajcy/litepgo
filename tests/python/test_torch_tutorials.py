"""Notebook infrastructure checks; lesson numerical checks run during the docs build."""

from pathlib import Path
import runpy
import sys

import pytest

nbformat = pytest.importorskip("nbformat")
pytest.importorskip("nbclient")
pytest.importorskip("nbconvert")
from nbclient.exceptions import CellExecutionError, CellTimeoutError

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "docs/scripts/tutorials.py"


@pytest.fixture
def pipeline():
    return runpy.run_path(str(SCRIPT))


def write_notebook(path, *cells):
    notebook = nbformat.v4.new_notebook(cells=list(cells), metadata={
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
    nbformat.write(notebook, path)
    return notebook


def test_seven_sources_are_clean_independent_lessons():
    paths = sorted((ROOT / "tutorials/torch").glob("[0-9][0-9]_*.ipynb"))
    assert len(paths) == 7
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        assert notebook.cells[0].cell_type == "markdown" and notebook.cells[0].source.startswith("# ")
        assert len([c for c in notebook.cells if c.cell_type == "markdown"]) >= 5
        code = [c for c in notebook.cells if c.cell_type == "code"]
        assert all(c.execution_count is None and c.outputs == [] for c in code)
        assert any("assert" in c.source for c in code)
        assert any("plt.show()" in c.source for c in code)
        assert "runpy" not in "\n".join(c.source for c in code)


def test_execute_export_images_and_source_unchanged(tmp_path, monkeypatch, pipeline):
    source = tmp_path / "01_sample.ipynb"
    write_notebook(source, nbformat.v4.new_markdown_cell("# Sample\n\n$$x^2$$"),
                   nbformat.v4.new_code_cell("%matplotlib inline\nimport matplotlib.pyplot as plt\n"
                                            "print('fresh kernel output')\nplt.plot([0, 1], [0, 1]); plt.show()"))
    asset = tmp_path / "mesh.veg"
    asset.write_text("test asset")
    monkeypatch.setitem(pipeline["render"].__globals__, "ASSETS", tmp_path)
    notebook = nbformat.read(source, as_version=4)
    notebook.metadata["tutorial_assets"] = [asset.name]
    checkpoint = tmp_path / "trained_theta.npy"
    checkpoint.write_bytes(b"test checkpoint")
    notebook.metadata["tutorial_files"] = [checkpoint.name]
    nbformat.write(notebook, source)
    original = source.read_bytes()
    cache = tmp_path / "cache/01_sample.ipynb"
    executed = pipeline["execute"](source, cache, 60)
    assert executed.metadata["tutorial_source_sha256"] == pipeline["fingerprint"](source)
    assert source.read_bytes() == original
    pages, downloads = tmp_path / "pages", tmp_path / "downloads"
    pipeline["render"](source, executed, pages, downloads, course="torch")
    markdown = (pages / "01_sample.md").read_text()
    assert "fresh kernel output" in markdown and "```python" in markdown and "$$x^2$$" in markdown
    assert "withBase('/tutorial-notebooks/torch/01_sample.ipynb')" in markdown
    assert "withBase('/tutorial-notebooks/torch/mesh.veg')" in markdown
    assert (downloads / asset.name).read_bytes() == asset.read_bytes()
    assert "withBase('/tutorial-notebooks/torch/trained_theta.npy')" in markdown
    assert (downloads / checkpoint.name).read_bytes() == checkpoint.read_bytes()
    assert list((pages / "assets/01_sample").glob("*.png"))
    assert (downloads / source.name).read_bytes() == original
    # Rendering may replace its own assets, but must preserve another lesson's output.
    unrelated = pages / "assets/02_other/keep.txt"
    unrelated.parent.mkdir()
    unrelated.write_text("keep")
    pipeline["render"](source, executed, pages, downloads, course="torch")
    assert unrelated.read_text() == "keep"


@pytest.mark.parametrize("source_code,timeout,error", [
    ("raise RuntimeError('lesson failure')", 30, CellExecutionError),
    ("import time\ntime.sleep(10)", 1, CellTimeoutError),
])
def test_errors_and_timeouts_fail_and_preserve_diagnostics(tmp_path, pipeline, source_code, timeout, error):
    source = tmp_path / "01_bad.ipynb"
    write_notebook(source, nbformat.v4.new_code_cell(source_code))
    cached = tmp_path / "failed.ipynb"
    with pytest.raises(error):
        pipeline["execute"](source, cached, timeout)
    failed = nbformat.read(cached, as_version=4)
    assert "tutorial_source_sha256" not in failed.metadata
    assert failed.cells[0].source == source_code


@pytest.mark.parametrize("changed", ["asset", "checkpoint", "source"])
def test_no_execute_rejects_changed_lesson_inputs(tmp_path, monkeypatch, pipeline, changed):
    sources = tmp_path / "tutorials/torch"
    sources.mkdir(parents=True)
    source = sources / "01_cached.ipynb"
    notebook = write_notebook(source, nbformat.v4.new_code_cell("print('ok')"))
    asset = tmp_path / "mesh.veg"
    asset.write_text("test asset")
    notebook.metadata["tutorial_assets"] = [asset.name]
    checkpoint = sources / "trained_theta.npy"
    checkpoint.write_bytes(b"test checkpoint")
    notebook.metadata["tutorial_files"] = [checkpoint.name]
    nbformat.write(notebook, source)
    cache = tmp_path / "build/torch-tutorials"
    cache.mkdir(parents=True)
    namespace = pipeline["main"].__globals__
    monkeypatch.setitem(namespace, "ASSETS", tmp_path)
    notebook.metadata["tutorial_source_sha256"] = pipeline["fingerprint"](source)
    nbformat.write(notebook, cache / source.name)
    monkeypatch.setitem(namespace, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--course", "torch", "--no-execute"])
    if changed == "asset":
        asset.write_text("changed asset")
    elif changed == "checkpoint":
        checkpoint.write_bytes(b"changed checkpoint")
    else:
        write_notebook(source, nbformat.v4.new_code_cell("print('changed')"))
    with pytest.raises(ValueError, match="stale or failed"):
        pipeline["main"]()
    assert len(list(cache.glob("*.ipynb"))) == 1
