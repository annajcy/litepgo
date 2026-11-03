from pathlib import Path
import numpy as np
import pytest
from litepgo import mesh

ASSETS = Path(__file__).resolve().parents[2] / 'assets'

@pytest.mark.parametrize('name,kind', [('shell.obj', 'obj'), ('box-tet.veg', 'veg'), ('box-hex.veg', 'veg')])
def test_real_asset_roundtrip(tmp_path, name, kind):
    read, write = getattr(mesh, 'read_' + kind), getattr(mesh, 'write_' + kind)
    data = read(ASSETS / name)
    assert data.positions.dtype == np.float64
    assert data.cells.shape[0] > 0
    path = tmp_path / name
    write(path, data)
    copy = read(path)
    if kind == "obj":
        np.testing.assert_allclose(copy.positions, data.positions, rtol=1e-15, atol=1e-15)
    else:
        np.testing.assert_array_equal(copy.positions, data.positions)
    np.testing.assert_array_equal(copy.cells, data.cells)

@pytest.mark.parametrize('kind', ['obj', 'veg', 'msh'])
def test_precision(tmp_path, kind):
    if kind == 'msh' and not mesh.has_gmsh:
        pytest.skip('Gmsh disabled')
    positions = np.array([[0.12345678901234567, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    data = mesh.TriangleMesh(positions, np.array([[0, 1, 2]], dtype=np.int32)) if kind == 'obj' else mesh.TetMesh(positions, np.array([[0, 1, 2, 3]], dtype=np.int32))
    path = tmp_path / ('mesh.' + kind)
    getattr(mesh, 'write_' + kind)(path, data)
    copy = getattr(mesh, 'read_' + kind)(path)
    np.testing.assert_allclose(copy.positions, positions, rtol=0, atol=1e-16)
    np.testing.assert_array_equal(copy.cells, data.cells)

def test_invalid_mesh():
    with pytest.raises(ValueError):
        mesh.TetMesh(np.zeros((4, 3)), np.array([[0, 1, 2, 4]], dtype=np.int32))
    with pytest.raises(ValueError):
        mesh.TriangleMesh(np.full((3, 3), np.nan), np.array([[0, 1, 2]], dtype=np.int32))

def test_negative_obj_indices(tmp_path):
    path = tmp_path / 'negative.obj'
    path.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n')
    np.testing.assert_array_equal(mesh.read_obj(path).cells, [[0, 1, 2]])

def test_veg_include_cycle(tmp_path):
    path = tmp_path / 'cycle.veg'
    path.write_text('*INCLUDE cycle.veg\n')
    with pytest.raises(RuntimeError, match='cycle'):
        mesh.read_veg(path)

def test_gmsh_disabled(tmp_path):
    if not mesh.has_gmsh:
        with pytest.raises(RuntimeError, match='LITEPGO_WITH_GMSH'):
            mesh.read_msh(tmp_path / 'absent.msh')

@pytest.mark.skipif(not mesh.has_cgal, reason="CGAL disabled")
@pytest.mark.parametrize(
    "extra_vertices, second_face, expected",
    [
        pytest.param([[0, -1, 0]], [1, 0, 3], False, id="shared-edge"),
        pytest.param(
            [[0.5, 0.5, -1], [0.5, 0.5, 1], [0.5, 1.5, 0]], [3, 4, 5], True, id="crossing"
        ),
        pytest.param([[0.5, 0.5, 0]], [1, 0, 3], True, id="shared-edge-overlap"),
    ],
)
def test_has_self_intersections(extra_vertices, second_face, expected):
    positions = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], *extra_vertices], dtype=np.float64)
    cells = np.array([[0, 1, 2], second_face], dtype=np.int32)
    assert mesh.has_self_intersections(mesh.TriangleMesh(positions, cells)) is expected


@pytest.mark.skipif(mesh.has_cgal, reason="CGAL enabled")
def test_has_self_intersections_requires_cgal():
    data = mesh.TriangleMesh(np.empty((0, 3)), np.empty((0, 3), dtype=np.int32))
    with pytest.raises(RuntimeError, match="LITEPGO_WITH_CGAL=ON"):
        mesh.has_self_intersections(data)


def test_isotropic_remesh():
    positions = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.2, 0.2, 0]])
    cells = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=np.int32)
    data = mesh.TriangleMesh(positions, cells)
    if not mesh.has_cgal:
        with pytest.raises(RuntimeError, match="LITEPGO_WITH_CGAL=ON"):
            mesh.isotropic_remesh(data, target_edge_length=0.8, num_iter=5)
        return

    result = mesh.isotropic_remesh(
        data, target_edge_length=0.8, num_iter=5, constrained_vertices=np.array([4], dtype=np.int64)
    )
    assert np.count_nonzero(np.all(result.positions == positions[4], axis=1)) == 1
    assert not mesh.has_self_intersections(result)
    np.testing.assert_array_equal(data.positions, positions)
    np.testing.assert_array_equal(data.cells, cells)
    with pytest.raises(ValueError, match="protected boundary edge"):
        mesh.isotropic_remesh(data, target_edge_length=0.5, num_iter=5)

@pytest.mark.parametrize('name', ['box-tet.veg', 'box-hex.veg'])
def test_veg_against_independent_text_parse(name):
    lines = [line.strip() for line in (ASSETS / name).read_text().splitlines() if line.strip() and not line.lstrip().startswith('#')]
    start = lines.index('*VERTICES')
    count = int(lines[start + 1].split()[0])
    vertices = [[float(x) for x in line.split()[1:4]] for line in lines[start + 2:start + 2 + count]]
    start = lines.index('*ELEMENTS')
    count = int(lines[start + 2].split()[0])
    cells = [[int(x) - 1 for x in line.split()[1:]] for line in lines[start + 3:start + 3 + count]]
    data = mesh.read_veg(ASSETS / name)
    np.testing.assert_array_equal(data.positions, vertices)
    np.testing.assert_array_equal(data.cells, cells)

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
    data = mesh.read_obj(ASSETS / 'shell.obj')
    np.testing.assert_allclose(data.positions, vertices, rtol=1e-15, atol=1e-15)
    np.testing.assert_array_equal(data.cells, faces)

@pytest.mark.skipif(not mesh.has_gmsh, reason='Gmsh disabled')
@pytest.mark.parametrize('name', ['box-tet.veg', 'box-hex.veg'])
def test_msh_real_asset_roundtrip(tmp_path, name):
    data = mesh.read_veg(ASSETS / name)
    path = tmp_path / 'volume.msh'
    mesh.write_msh(path, data)
    copy = mesh.read_msh(path)
    np.testing.assert_allclose(copy.positions, data.positions, rtol=1e-15, atol=1e-15)
    np.testing.assert_array_equal(copy.cells, data.cells)

@pytest.mark.skipif(not mesh.has_gmsh, reason='Gmsh disabled')
def test_msh_sparse_tags(tmp_path):
    path = tmp_path / 'sparse.msh'
    path.write_text('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n4\n40 0 0 0\n10 1 0 0\n80 0 1 0\n25 0 0 1\n$EndNodes\n$Elements\n1\n77 4 0 40 10 80 25\n$EndElements\n')
    data = mesh.read_msh(path)
    np.testing.assert_array_equal(data.positions[data.cells[0]], [[0,0,0], [1,0,0], [0,1,0], [0,0,1]])

@pytest.mark.parametrize('text', ['*VERTICES\n1 3 0 0\n1 0 0 junk\n', '*VERTICES\n2 3 0 0\n1 0 0 0\n'])
def test_bad_veg(tmp_path, text):
    path = tmp_path / 'bad.veg'
    path.write_text(text)
    with pytest.raises(RuntimeError):
        mesh.read_veg(path)

@pytest.mark.skipif(not mesh.has_gmsh, reason='Gmsh disabled')
def test_msh_invalid_output_preserves_file(tmp_path):
    path = tmp_path / 'keep.obj'
    path.write_text('keep')
    data = mesh.read_veg(ASSETS / 'box-tet.veg')
    with pytest.raises(ValueError):
        mesh.write_msh(path, data)
    assert path.read_text() == 'keep'
    empty = mesh.TetMesh(np.empty((0, 3)), np.empty((0, 4), dtype=np.int32))
    with pytest.raises(ValueError):
        mesh.write_msh(tmp_path / 'empty.msh', empty)

@pytest.mark.skipif(not mesh.has_gmsh, reason='Gmsh disabled')
def test_msh_rejects_surface_only(tmp_path):
    path = tmp_path / 'surface.msh'
    path.write_text('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n3\n1 0 0 0\n2 1 0 0\n3 0 1 0\n$EndNodes\n$Elements\n1\n1 2 0 1 2 3\n$EndElements\n')
    with pytest.raises(RuntimeError, match='tet4 or hex8'):
        mesh.read_msh(path)

def test_veg_include(tmp_path):
    (tmp_path / 'vertices.inc').write_text('*VERTICES\n4 3 0 0\n1 0 0 0\n2 1 0 0\n3 0 1 0\n4 0 0 1\n')
    path = tmp_path / 'mesh.veg'
    path.write_text('*INCLUDE vertices.inc\n*ELEMENTS\nTET\n1 4 0\n1 1 2 3 4\n')
    np.testing.assert_array_equal(mesh.read_veg(path).cells, [[0, 1, 2, 3]])

@pytest.mark.skipif(not mesh.has_gmsh, reason='Gmsh disabled')
def test_msh_rejects_mixed_volume_cells(tmp_path):
    path = tmp_path / 'mixed.msh'
    path.write_text('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n8\n1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n5 0 0 1\n6 1 0 1\n7 1 1 1\n8 0 1 1\n$EndNodes\n$Elements\n2\n1 4 0 1 2 4 5\n2 5 0 1 2 3 4 5 6 7 8\n$EndElements\n')
    with pytest.raises(RuntimeError, match='homogeneous'):
        mesh.read_msh(path)

def _enclosed_volume(surface):
    p, f = surface.positions, surface.cells
    return np.einsum('ij,ij->i', p[f[:, 0]], np.cross(p[f[:, 1]], p[f[:, 2]])).sum() / 6

@pytest.mark.parametrize('name', ['box-tet.veg', 'box-hex.veg'])
def test_extract_surface_is_closed_and_outward(name):
    data = mesh.read_veg(ASSETS / name)
    surface = mesh.extract_surface(data)
    edges = {}
    for a, b, c in surface.cells:
        for edge in ((a, b), (b, c), (c, a)):
            edges[edge] = edges.get(edge, 0) + 1
    assert all(count == 1 and edges.get((b, a)) == 1 for (a, b), count in edges.items())
    extent = data.positions.max(axis=0) - data.positions.min(axis=0)
    assert _enclosed_volume(surface) == pytest.approx(np.prod(extent), rel=1e-12)
    # Interior vertices are dropped; kept ones are copied unchanged in original order.
    on_surface = [np.flatnonzero((data.positions == p).all(axis=1))[0] for p in surface.positions]
    assert len(surface.positions) < len(data.positions)
    assert np.all(np.diff(on_surface) > 0)

def test_extract_surface_single_cells():
    tet = mesh.TetMesh(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [5, 5, 5.]]), np.array([[0, 1, 2, 3]], dtype=np.int32))
    surface = mesh.extract_surface(tet)
    assert surface.cells.shape == (4, 3) and len(surface.positions) == 4
    assert _enclosed_volume(surface) == pytest.approx(1 / 6)
    corners = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1.]])
    surface = mesh.extract_surface(mesh.HexMesh(2 * corners, np.arange(8, dtype=np.int32)[None]))
    assert surface.cells.shape == (12, 3)
    assert _enclosed_volume(surface) == pytest.approx(8)

def test_extract_surface_rejects_overshared_face():
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1], [1, 1, 1.]])
    cells = np.array([[0, 1, 2, 3], [0, 2, 1, 4], [0, 1, 2, 5]], dtype=np.int32)
    with pytest.raises(ValueError, match='more than two'):
        mesh.extract_surface(mesh.TetMesh(positions, cells))

def test_topology_mirrors_cpp():
    topology = mesh.TetTopology(5, np.array([[0, 1, 2, 3], [0, 2, 1, 4]], dtype=np.int32))
    assert mesh.QuadTopology is mesh.TetTopology
    assert (topology.num_vertices, topology.num_cells, topology.vertices_per_cell) == (5, 2, 4)
    np.testing.assert_array_equal(topology.cell_vertices(1), [0, 2, 1, 4])
    with pytest.raises(IndexError):
        topology.cell_vertices(2)
    with pytest.raises(ValueError):
        mesh.TetTopology(3, np.array([[0, 1, 2, 3]], dtype=np.int32))
    positions = np.arange(15.0).reshape(5, 3)
    data = mesh.TetMesh(positions, topology)
    np.testing.assert_array_equal(data.cells, topology.cells)
    np.testing.assert_array_equal(data.topology.cells, topology.cells)
    with pytest.raises(ValueError):
        mesh.TetMesh(positions[:4], topology)

def test_triangle_mid_edge_stencils():
    # Two triangles sharing edge (1, 2).
    topology = mesh.TriangleTopology(4, np.array([[0, 1, 2], [2, 1, 3]], dtype=np.int32))
    stencils = mesh.make_triangle_mid_edge_stencils(topology)
    assert [list(v) for v in stencils.vertices] == [[0, 1, 2, 3], [2, 1, 3, 0]]
    np.testing.assert_array_equal(stencils.opposite, [[3, -1, -1], [-1, -1, 3]])
