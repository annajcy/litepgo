"""Run with a Python environment containing pypgo and this litepgo build."""
from pathlib import Path
import tempfile
import numpy as np
from litepgo import io
from pypgo.io import mesh as reference

assets = Path(__file__).resolve().parents[1] / 'assets'
with tempfile.TemporaryDirectory() as directory:
    for name, kind in [('shell.obj', 'obj'), ('box-tet.veg', 'veg'), ('box-hex.veg', 'veg')]:
        mesh = getattr(io, 'read_' + kind)(assets / name)
        expected = getattr(reference, 'read_' + kind)(str(assets / name))
        np.testing.assert_allclose(mesh.positions, expected.vertices, rtol=1e-15, atol=1e-15)
        np.testing.assert_array_equal(mesh.cells, expected.elements)
        output = Path(directory) / name
        getattr(io, 'write_' + kind)(output, mesh)
        actual = getattr(reference, 'read_' + kind)(str(output))
        np.testing.assert_allclose(mesh.positions, actual.vertices, rtol=1e-15, atol=1e-15)
        np.testing.assert_array_equal(mesh.cells, actual.elements)
        getattr(reference, 'write_' + kind)(str(output), expected)
        actual = getattr(io, 'read_' + kind)(output)
        np.testing.assert_allclose(mesh.positions, actual.positions, rtol=1e-15, atol=1e-15)
        np.testing.assert_array_equal(mesh.cells, actual.cells)
        print(name, 'bidirectional libpgo differential passed')
    if io.has_gmsh and reference.has_gmsh():
        mesh = io.read_veg(assets / 'box-tet.veg')
        output = Path(directory) / 'tet.msh'
        io.write_msh(output, mesh)
        expected = reference.read_msh(str(output))
        np.testing.assert_allclose(mesh.positions, expected.vertices, rtol=1e-15, atol=1e-15)
        np.testing.assert_array_equal(mesh.cells, expected.elements)
        print('tet MSH litepgo writer / libpgo reader differential passed')

# Independent Gmsh serialization covers both binary MSH versions.
if io.has_gmsh:
    import gmsh
    with tempfile.TemporaryDirectory() as directory:
        mesh = io.read_veg(assets / 'box-tet.veg')
        source = Path(directory) / 'source.msh'
        io.write_msh(source, mesh)
        for version in (2.2, 4.1):
            output = Path(directory) / f'binary-{version}.msh'
            gmsh.initialize()
            try:
                gmsh.open(str(source))
                gmsh.option.setNumber('Mesh.MshFileVersion', version)
                gmsh.option.setNumber('Mesh.Binary', 1)
                gmsh.write(str(output))
            finally:
                gmsh.finalize()
            actual = io.read_msh(output)
            np.testing.assert_array_equal(actual.positions, mesh.positions)
            np.testing.assert_array_equal(actual.cells, mesh.cells)
            print('Binary MSH', version, 'Gmsh writer / litepgo reader passed')
