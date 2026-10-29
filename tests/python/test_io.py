from pathlib import Path
import numpy as np
import pytest
from litepgo import io

ASSETS = Path(__file__).resolve().parents[2] / 'assets'

@pytest.mark.parametrize('name,kind', [('shell.obj', 'obj'), ('box-tet.veg', 'veg'), ('box-hex.veg', 'veg')])
def test_real_asset_roundtrip(tmp_path, name, kind):
    read, write = getattr(io, 'read_' + kind), getattr(io, 'write_' + kind)
    mesh = read(ASSETS / name)
    assert mesh.positions.dtype == np.float64
    assert mesh.cells.shape[0] > 0
    path = tmp_path / name
    write(path, mesh)
    copy = read(path)
    if kind == "obj":
        np.testing.assert_allclose(copy.positions, mesh.positions, rtol=1e-15, atol=1e-15)
    else:
        np.testing.assert_array_equal(copy.positions, mesh.positions)
    np.testing.assert_array_equal(copy.cells, mesh.cells)

@pytest.mark.parametrize('kind', ['obj', 'veg', 'msh'])
def test_precision(tmp_path, kind):
    if kind == 'msh' and not io.has_gmsh:
        pytest.skip('Gmsh disabled')
    positions = np.array([[0.12345678901234567, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    mesh = io.TriangleMesh(positions, np.array([[0, 1, 2]], dtype=np.int32)) if kind == 'obj' else io.TetMesh(positions, np.array([[0, 1, 2, 3]], dtype=np.int32))
    path = tmp_path / ('mesh.' + kind)
    getattr(io, 'write_' + kind)(path, mesh)
    copy = getattr(io, 'read_' + kind)(path)
    np.testing.assert_allclose(copy.positions, positions, rtol=0, atol=1e-16)
    np.testing.assert_array_equal(copy.cells, mesh.cells)

def test_invalid_mesh():
    with pytest.raises(ValueError):
        io.TetMesh(np.zeros((4, 3)), np.array([[0, 1, 2, 4]], dtype=np.int32))
    with pytest.raises(ValueError):
        io.TriangleMesh(np.full((3, 3), np.nan), np.array([[0, 1, 2]], dtype=np.int32))

def test_negative_obj_indices(tmp_path):
    path = tmp_path / 'negative.obj'
    path.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n')
    np.testing.assert_array_equal(io.read_obj(path).cells, [[0, 1, 2]])

def test_veg_include_cycle(tmp_path):
    path = tmp_path / 'cycle.veg'
    path.write_text('*INCLUDE cycle.veg\n')
    with pytest.raises(RuntimeError, match='cycle'):
        io.read_veg(path)

def test_gmsh_disabled(tmp_path):
    if not io.has_gmsh:
        with pytest.raises(RuntimeError, match='LITEPGO_WITH_GMSH'):
            io.read_msh(tmp_path / 'absent.msh')

@pytest.mark.parametrize('name', ['box-tet.veg', 'box-hex.veg'])
def test_veg_against_independent_text_parse(name):
    lines = [line.strip() for line in (ASSETS / name).read_text().splitlines() if line.strip() and not line.lstrip().startswith('#')]
    start = lines.index('*VERTICES')
    count = int(lines[start + 1].split()[0])
    vertices = [[float(x) for x in line.split()[1:4]] for line in lines[start + 2:start + 2 + count]]
    start = lines.index('*ELEMENTS')
    count = int(lines[start + 2].split()[0])
    cells = [[int(x) - 1 for x in line.split()[1:]] for line in lines[start + 3:start + 3 + count]]
    mesh = io.read_veg(ASSETS / name)
    np.testing.assert_array_equal(mesh.positions, vertices)
    np.testing.assert_array_equal(mesh.cells, cells)

def test_obj_against_independent_text_parse():
    vertices, faces = [], []
    for line in (ASSETS / 'shell.obj').read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == 'v':
            vertices.append([float(x) for x in parts[1:4]])
        elif parts[0] == 'f':
            assert len(parts) == 4
            faces.append([int(x.split('/')[0]) - 1 for x in parts[1:]])
    mesh = io.read_obj(ASSETS / 'shell.obj')
    np.testing.assert_allclose(mesh.positions, vertices, rtol=1e-15, atol=1e-15)
    np.testing.assert_array_equal(mesh.cells, faces)

@pytest.mark.skipif(not io.has_gmsh, reason='Gmsh disabled')
@pytest.mark.parametrize('name', ['box-tet.veg', 'box-hex.veg'])
def test_msh_real_asset_roundtrip(tmp_path, name):
    mesh = io.read_veg(ASSETS / name)
    path = tmp_path / 'volume.msh'
    io.write_msh(path, mesh)
    copy = io.read_msh(path)
    np.testing.assert_allclose(copy.positions, mesh.positions, rtol=1e-15, atol=1e-15)
    np.testing.assert_array_equal(copy.cells, mesh.cells)

@pytest.mark.skipif(not io.has_gmsh, reason='Gmsh disabled')
def test_msh_sparse_tags(tmp_path):
    path = tmp_path / 'sparse.msh'
    path.write_text('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n4\n40 0 0 0\n10 1 0 0\n80 0 1 0\n25 0 0 1\n$EndNodes\n$Elements\n1\n77 4 0 40 10 80 25\n$EndElements\n')
    mesh = io.read_msh(path)
    np.testing.assert_array_equal(mesh.positions[mesh.cells[0]], [[0,0,0], [1,0,0], [0,1,0], [0,0,1]])

@pytest.mark.parametrize('text', ['*VERTICES\n1 3 0 0\n1 0 0 junk\n', '*VERTICES\n2 3 0 0\n1 0 0 0\n'])
def test_bad_veg(tmp_path, text):
    path = tmp_path / 'bad.veg'
    path.write_text(text)
    with pytest.raises(RuntimeError):
        io.read_veg(path)

@pytest.mark.skipif(not io.has_gmsh, reason='Gmsh disabled')
def test_msh_invalid_output_preserves_file(tmp_path):
    path = tmp_path / 'keep.obj'
    path.write_text('keep')
    mesh = io.read_veg(ASSETS / 'box-tet.veg')
    with pytest.raises(ValueError):
        io.write_msh(path, mesh)
    assert path.read_text() == 'keep'
    empty = io.TetMesh(np.empty((0, 3)), np.empty((0, 4), dtype=np.int32))
    with pytest.raises(ValueError):
        io.write_msh(tmp_path / 'empty.msh', empty)

@pytest.mark.skipif(not io.has_gmsh, reason='Gmsh disabled')
def test_msh_rejects_surface_only(tmp_path):
    path = tmp_path / 'surface.msh'
    path.write_text('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n3\n1 0 0 0\n2 1 0 0\n3 0 1 0\n$EndNodes\n$Elements\n1\n1 2 0 1 2 3\n$EndElements\n')
    with pytest.raises(RuntimeError, match='tet4 or hex8'):
        io.read_msh(path)

def test_veg_include(tmp_path):
    (tmp_path / 'vertices.inc').write_text('*VERTICES\n4 3 0 0\n1 0 0 0\n2 1 0 0\n3 0 1 0\n4 0 0 1\n')
    path = tmp_path / 'mesh.veg'
    path.write_text('*INCLUDE vertices.inc\n*ELEMENTS\nTET\n1 4 0\n1 1 2 3 4\n')
    np.testing.assert_array_equal(io.read_veg(path).cells, [[0, 1, 2, 3]])

@pytest.mark.skipif(not io.has_gmsh, reason='Gmsh disabled')
def test_msh_rejects_mixed_volume_cells(tmp_path):
    path = tmp_path / 'mixed.msh'
    path.write_text('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n8\n1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n5 0 0 1\n6 1 0 1\n7 1 1 1\n8 0 1 1\n$EndNodes\n$Elements\n2\n1 4 0 1 2 4 5\n2 5 0 1 2 3 4 5 6 7 8\n$EndElements\n')
    with pytest.raises(RuntimeError, match='homogeneous'):
        io.read_msh(path)
