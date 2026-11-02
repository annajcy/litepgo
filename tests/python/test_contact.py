import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

import litepgo
from litepgo import _core, dof, energy, mesh

from litepgo import contact  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
# The extension lives in build/<preset>/python/litepgo; the examples in build/<preset>/examples/contact.
EXAMPLES = Path(_core.__file__).parents[2] / "examples" / "contact"


def two_triangles(height):
    """Triangle A in z = 0 and triangle B pointing vertex 3 down at (2, 2, height)."""
    positions = np.array(
        [[0, 0, 0], [10, 0, 0], [0, 10, 0], [2, 2, height], [6, 2, height + 4], [2, 6, height + 4.0]]
    )
    return mesh.TriangleMesh(positions, np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32))


def central_gradient(f, x, step=1e-6):
    return np.array([(f(x + step * e) - f(x - step * e)) / (2 * step) for e in np.eye(len(x))])


def test_build_info_reports_ipc_toolkit():
    assert hasattr(contact, "IPCTKContactEnergy") == bool(litepgo.build_info()["ipc_toolkit_version"])


def test_barrier_mirrors_cpp():
    surface = two_triangles(0.5)
    barrier = contact.IPCContactEnergy(surface, contact.IPCContactOptions(dhat=1.0), contact.SpatialHash())
    assert isinstance(barrier, energy.Energy) and not barrier.has_fixed_pattern
    assert barrier.num_vertices == 6 and barrier.num_dofs == 18
    np.testing.assert_array_equal(barrier.rest_positions, surface.positions)
    assert barrier.options.dhat == 1.0 and barrier.options.min_distance == 0.0

    x = surface.positions.ravel()
    assert barrier.num_active_pairs(x) == 1
    assert barrier.min_active_distance(x) == pytest.approx(0.5, abs=1e-14)
    assert barrier.eval_value(x) > 0
    np.testing.assert_allclose(barrier.eval_gradient(x), central_gradient(barrier.eval_value, x), atol=1e-6)

    assert barrier.update_pattern(x)
    assert barrier.eval_hessian(x).nnz == 144
    assert np.linalg.eigvalsh(barrier.eval_projected_hessian(x).toarray()).min() > -1e-10


def test_barrier_rejects_infeasible_states():
    surface = two_triangles(0.5)
    with pytest.raises(ValueError):
        contact.IPCContactEnergy(surface, contact.IPCContactOptions(), contact.SpatialHash())  # dhat unset
    thick = contact.IPCContactEnergy(surface, contact.IPCContactOptions(dhat=1.0, min_distance=0.6), contact.SpatialHash())
    with pytest.raises(ValueError, match="min_distance"):
        thick.eval_value(surface.positions.ravel())


def test_planes_and_feasible_steps():
    surface = two_triangles(0.5)
    triangle = mesh.TriangleMesh(surface.positions[3:], np.array([[0, 1, 2]], dtype=np.int32))
    ground = contact.Plane(point=np.zeros(3), normal=np.array([0, 0, 1.0]))
    barrier = contact.IPCContactEnergy(triangle, contact.IPCContactOptions(dhat=1.0), contact.SpatialHash(), [ground])
    x = triangle.positions.ravel()
    assert barrier.num_active_pairs(x) == 1
    down = np.tile([0, 0, -1.0], 3)
    assert 0 < barrier.max_feasible_step(x, down) < 0.5
    assert barrier.max_feasible_step(x, -down) == 1.0


def test_barrier_composes_with_a_surface_kinematics():
    barrier = contact.IPCContactEnergy(two_triangles(0.5), contact.IPCContactOptions(dhat=1.0), contact.SpatialHash())
    # A is a static obstacle; B's vertices take simulation vertices 0, 1, 2.
    kinematics = dof.SurfaceKinematics(barrier.rest_positions, sp.eye(18, 9, k=-9, format="csc"))
    np.testing.assert_array_equal(kinematics.displacement_map.toarray(), sp.eye(18, 9, k=-9).toarray())
    mapped = energy.AffineMappedEnergy(barrier, kinematics.displacement_map, kinematics.rest_positions.ravel())
    u = np.zeros(9)
    u[2] = -0.2
    x = barrier.rest_positions.ravel().copy()
    x[11] -= 0.2
    assert mapped.eval_value(u) == barrier.eval_value(x)
    np.testing.assert_array_equal(mapped.eval_gradient(u), barrier.eval_gradient(x)[9:])


def parse_results(stdout):
    values = dict(item.split("=") for item in stdout.strip().split(", "))
    exact = ("converged", "iterations", "active pairs")
    return {key: int(value) if key in exact else float(value) for key, value in values.items()}


def test_python_box_stack_matches_cpp(tmp_path):
    executable = EXAMPLES / "tet_p1_box_stack"
    if not executable.exists():
        pytest.skip(f"{executable} is not built")
    script = REPO / "examples" / "contact" / "tet_p1_box_stack.py"
    asset = REPO / "assets/box-tet.veg"
    cpp_out, py_out = tmp_path / "cpp.veg", tmp_path / "py.veg"
    cpp = subprocess.run([executable, asset, cpp_out], check=True, capture_output=True, text=True)
    py = subprocess.run([sys.executable, script, asset, py_out], check=True, capture_output=True, text=True)
    cpp_results, py_results = parse_results(cpp.stdout), parse_results(py.stdout)
    assert py_results.keys() == cpp_results.keys()
    for key, value in cpp_results.items():
        if key in ("converged", "iterations", "active pairs"):
            assert py_results[key] == value, key
        elif key == "free residual":
            assert value < 1e-8 and py_results[key] < 1e-8
        else:
            assert py_results[key] == pytest.approx(value, rel=1e-5), key
    cpp_mesh, py_mesh = mesh.read_veg(cpp_out), mesh.read_veg(py_out)
    np.testing.assert_array_equal(py_mesh.cells, cpp_mesh.cells)
    np.testing.assert_allclose(py_mesh.positions, cpp_mesh.positions, rtol=0, atol=1e-10)


@pytest.mark.skipif(not hasattr(contact, "IPCTKContactEnergy"), reason="toolkit backend disabled")
def test_independent_native_matches_toolkit_original_sum():
    # Compare independent implementations at generic closest features, including
    # repeated lower-dimensional features. Projected Hessians need not match.
    surface = two_triangles(0.5)
    native = contact.IPCContactEnergy(surface, contact.IPCContactOptions(dhat=3, min_distance=0.1), contact.SpatialHash())
    toolkit = contact.IPCTKContactEnergy(surface, contact.IPCTKContactOptions(dhat=3, min_distance=0.1))
    x = surface.positions.ravel().copy()
    x[12:15] += [0.3, -0.2, 0.1]
    for backend in (native, toolkit):
        backend.update_pattern(x)
    assert native.eval_value(x) == pytest.approx(toolkit.eval_value(x), rel=1e-12)
    np.testing.assert_allclose(native.eval_gradient(x), toolkit.eval_gradient(x), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(native.eval_hessian(x).toarray(), toolkit.eval_hessian(x).toarray(), rtol=1e-9, atol=1e-9)
    with pytest.raises(TypeError):
        contact.IPCContactEnergy(surface, contact.IPCTKContactOptions(dhat=1), contact.SpatialHash())


@pytest.mark.parametrize("backend_factory", [contact.SpatialHash, lambda: contact.SpatialHash(cell_size=0.5)])
def test_native_broad_phase_injection_and_lifetime(backend_factory):
    import gc

    surface = two_triangles(0.5)
    options = contact.IPCContactOptions(dhat=1)
    with pytest.raises(TypeError):
        contact.IPCContactEnergy(surface, options)
    with pytest.raises((TypeError, ValueError)):
        contact.IPCContactEnergy(surface, options, None)
    with pytest.raises(TypeError):
        contact.IPCContactEnergy(surface, options, [])
    backend = backend_factory()
    assert isinstance(backend, contact.BroadPhase)
    with pytest.raises(TypeError):
        contact.BroadPhase()
    energies = [contact.IPCContactEnergy(surface, options, contact.SpatialHash()),
                contact.IPCContactEnergy(surface, options, backend, []),
                contact.IPCContactEnergy(surface, options, broad_phase=backend)]
    del backend
    gc.collect()
    x = surface.positions.ravel()
    dx = np.zeros_like(x)
    dx[11] = -2
    for barrier in energies:
        barrier.update_pattern(x)
        assert barrier.eval_value(x) == energies[0].eval_value(x)
        np.testing.assert_array_equal(barrier.eval_gradient(x), energies[0].eval_gradient(x))
        np.testing.assert_array_equal(barrier.eval_hessian(x).toarray(), energies[0].eval_hessian(x).toarray())
        assert barrier.max_feasible_step(x, dx) == energies[0].max_feasible_step(x, dx)
    del energies[1]
    gc.collect()
    assert energies[1].eval_value(x) == energies[0].eval_value(x)
    with pytest.raises(TypeError):
        contact.IPCContactEnergy(surface, options, broad_phase=object())
    if hasattr(contact, "IPCTKContactEnergy"):
        with pytest.raises(TypeError):
            contact.IPCTKContactEnergy(surface, contact.IPCTKContactOptions(dhat=1),
                                       broad_phase=contact.SpatialHash())


@pytest.mark.skipif(not hasattr(contact, "IPCTKContactEnergy"), reason="toolkit backend disabled")
@pytest.mark.parametrize("positions", [
    [[0, 0, 0], [2, 0, 0], [0, 2, 0], [-0.2, -0.3, 0.4], [-1, -2, 1], [-2, -1, 1]],
    [[0, 0, 0], [2, 0, 0], [0, 2, 0], [0.7, -0.2, 0.4], [2, -2, 1], [-2, -1, 1]],
    [[0, 0, 0], [2, 0, 0], [0, 2, 0], [0.4, 0.3, 0.4], [2, 2, 1], [-1, 2, 1]],
    [[-1, 0, 0], [1, 0, 0], [0, -4, -2], [-0.7, -0.004, 0.2], [0.8, 0.008, 0.2], [0, 4, 2]],
    [[-1, 0, 0], [1, 0, 0], [0, -4, -2], [-0.7, 0, 0.2], [0.8, 0, 0.2], [0, 4, 2]],
])
def test_analytic_features_and_mollifier_match_toolkit(positions):
    surface = mesh.TriangleMesh(np.array(positions, dtype=float), np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32))
    native = contact.IPCContactEnergy(surface, contact.IPCContactOptions(dhat=0.7, min_distance=0.05), contact.SpatialHash())
    toolkit = contact.IPCTKContactEnergy(surface, contact.IPCTKContactOptions(dhat=0.7, min_distance=0.05))
    x = surface.positions.ravel()
    for barrier in (native, toolkit):
        barrier.update_pattern(x)
    assert native.eval_value(x) == pytest.approx(toolkit.eval_value(x), rel=1e-11, abs=1e-12)
    np.testing.assert_allclose(native.eval_gradient(x), toolkit.eval_gradient(x), rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(native.eval_hessian(x).toarray(), toolkit.eval_hessian(x).toarray(), rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("cell_size", [0, -1, float("inf"), float("nan")])
def test_spatial_hash_rejects_invalid_cell_size(cell_size):
    with pytest.raises(ValueError):
        contact.SpatialHash(cell_size)
