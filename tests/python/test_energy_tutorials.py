"""Check each Energy lesson's sources before executing them in their own kernels."""

from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")


@pytest.mark.parametrize("lesson", ["01_spring_chain", "02_spring_response"])
def test_energy_notebooks_share_the_lesson(lesson):
    folder = Path(__file__).resolve().parents[2] / "tutorials/energy"
    python = nbformat.read(folder / f"{lesson}_python.ipynb", as_version=4)
    cpp = nbformat.read(folder / f"{lesson}_cpp.ipynb", as_version=4)

    for notebook, kernel in ((python, "python3"), (cpp, "litepgo-cpp20")):
        nbformat.validate(notebook)
        assert notebook.metadata.kernelspec.name == kernel
        assert all(
            cell.execution_count is None and cell.outputs == []
            for cell in notebook.cells
            if cell.cell_type == "code"
        )

    for python_cell, cpp_cell in zip(python.cells, cpp.cells, strict=True):
        assert python_cell.cell_type == cpp_cell.cell_type
        if python_cell.cell_type != "code":
            assert python_cell.source == cpp_cell.source
