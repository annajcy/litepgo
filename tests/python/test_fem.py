import gc
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from litepgo import _core, dof, energy, fem, mesh

ASSETS = Path(__file__).resolve().parents[2] / "assets"
# The extension lives in build/<preset>/python/litepgo; the examples in build/<preset>/examples/fem.
EXAMPLES = Path(_core.__file__).parents[2] / "examples" / "fem"

NEO_HOOKEAN_THETA = np.array([2.0, 3.0])


def test_templates_mirror_cpp_arguments():
    Energy = fem.VolumetricFemEnergy[fem.NeoHookean, fem.HexQ1Basis, fem.GaussLegendre3D[2]]
    assert Energy.__name__ == "VolumetricFemEnergy[NeoHookean, HexQ1Basis, GaussLegendre3D[2]]"
    assert issubclass(Energy, energy.Energy) and issubclass(Energy, fem.VolumetricFemEnergy)
    assert fem.InterpolatedField3D[fem.TetP1Basis] is fem.InterpolatedField3D[(fem.TetP1Basis,)]
    assert issubclass(fem.InterpolatedField3D[fem.TetP1Basis], fem.Field3D)
    assert fem.GaussLegendre3D[4].num_points == 64
    with pytest.raises(TypeError, match=r"available: .*GaussLegendre3D\[2\]"):
        fem.GaussLegendre3D[3]
    with pytest.raises(TypeError, match="VolumetricFemEnergy"):
        fem.VolumetricFemEnergy[fem.NeoHookean, fem.TetP1Basis, fem.GaussLegendre3D[2]]
    with pytest.raises(TypeError):
        fem.TetP1Basis()


@pytest.mark.parametrize(
    "basis,dim", [(fem.TriangleP1Basis, 2), (fem.QuadHermiteBasis, 2), (fem.TetP1Basis, 3), (fem.HexQ1Basis, 3)]
)
def test_lagrange_like_bases_partition_unity(basis, dim):
    rng = np.random.default_rng(0)
    xi = 0.3 * rng.random(dim)
    N = basis.eval_N(xi)
    assert N.shape == (basis.num_basis_functions,)
    if basis is not fem.QuadHermiteBasis:  # Hermite value functions alone sum to one
        assert N.sum() == pytest.approx(1.0)
    h = 1e-6
    for k in range(dim):
        step = np.zeros(dim)
        step[k] = h
        fd = (basis.eval_N(xi + step) - basis.eval_N(xi - step)) / (2 * h)
        np.testing.assert_allclose(basis.eval_dN_dxi(xi)[:, k], fd, atol=1e-8)


def test_basis_reference_nodes_interpolate():
    for i in range(fem.TetP1Basis.num_basis_functions):
        np.testing.assert_allclose(fem.TetP1Basis.eval_N(fem.TetP1Basis.reference_node(i)), np.eye(4)[i], atol=1e-15)


@pytest.mark.parametrize(
    "quadrature,volume",
    [
        (fem.TetCentroidQuadrature, 1 / 6),
        (fem.TriangleCentroidQuadrature, 1 / 2),
        (fem.GaussLegendre2D[2], 1.0),
        (fem.GaussLegendre3D[4], 1.0),
    ],
)
def test_quadrature_weights(quadrature, volume):
    assert quadrature.points().shape[0] == quadrature.num_points
    assert quadrature.weights().sum() == pytest.approx(volume)


def test_fields():
    coeffs = np.array([fem.HexQ1Basis.reference_node(v) for v in range(8)]) * [2.0, 1.0, 3.0]
    X = fem.InterpolatedField3D[fem.HexQ1Basis](coeffs)
    xi = np.array([0.2, 0.5, 0.9])
    np.testing.assert_allclose(X.eval_value(xi), [0.4, 0.5, 2.7])
    np.testing.assert_allclose(X.eval_dvalue_dxi(xi), np.diag([2.0, 1.0, 3.0]), atol=1e-15)
    assert (X.num_components, X.num_dofs) == (3, 24)
    constant = fem.IdentityField3D(np.array([1.0, 2.0]))
    np.testing.assert_array_equal(constant.eval_value(xi), [1.0, 2.0])
    with pytest.raises(ValueError):
        fem.InterpolatedField3D[fem.HexQ1Basis](coeffs[:7])


def test_deformation_gradient_keeps_its_fields_alive():
    def make():
        nodes = np.array([fem.TetP1Basis.reference_node(v) for v in range(4)])
        u = fem.InterpolatedField3D[fem.TetP1Basis](0.1 * nodes)
        X = fem.InterpolatedField3D[fem.TetP1Basis](nodes)
        return fem.DeformationGradientField3D(u, X)

    F = make()
    gc.collect()
    np.testing.assert_allclose(F.eval_value(np.full(3, 0.25)), (1.1 * np.eye(3)).ravel(order="F"))


def test_materials_pointwise():
    I = np.eye(3)
    assert fem.NeoHookean.eval_W(I, I, NEO_HOOKEAN_THETA) == pytest.approx(0.0, abs=1e-15)
    F = I + 0.1 * np.arange(9.0).reshape(3, 3) / 9
    P = fem.NeoHookean.eval_dW_dF(F, I, NEO_HOOKEAN_THETA)
    h = 1e-6
    for i in range(3):
        for j in range(3):
            dF = np.zeros((3, 3))
            dF[i, j] = h
            fd = (fem.NeoHookean.eval_W(F + dF, I, NEO_HOOKEAN_THETA) - fem.NeoHookean.eval_W(F - dF, I, NEO_HOOKEAN_THETA)) / (2 * h)
            assert P[i, j] == pytest.approx(fd, abs=1e-8)
    assert fem.NeoHookean.eval_feasible_step(I, I, NEO_HOOKEAN_THETA, -2 * I, 0 * I, 0 * NEO_HOOKEAN_THETA) < 0.5
    a = np.array([[1.0, 0.2], [0.2, 1.5]])
    assert fem.KoiterStVK.eval_W(a, 0 * a, a, 0 * a, 0.1, np.array([1.0, 2.0])) == 0.0
    with pytest.raises(ValueError):
        fem.NeoHookean.eval_W(I, I, np.array([1.0, -1.0]))


def rest_fields(XField, rest_map, positions):
    return [XField(rest_map.gather(e, positions)) for e in range(rest_map.num_elements)]


def constant_fields(value, count):
    return [fem.IdentityField3D(np.asarray(value, dtype=float).ravel())] * count


def volumetric_energy(data, basis, quadrature, dofs_per_vertex, theta_fields=None):
    rest_map = dof.VertexDofMap(data.topology, 3)
    X_fields = rest_fields(fem.InterpolatedField3D[type_of_rest_basis(data)], rest_map, data.positions.ravel())
    n = data.topology.num_cells
    Energy = fem.VolumetricFemEnergy[fem.NeoHookean, basis, quadrature]
    return Energy(
        dof.VertexDofMap(data.topology, dofs_per_vertex),
        X_fields,
        constant_fields(np.eye(3), n),
        theta_fields or constant_fields(NEO_HOOKEAN_THETA, n),
    )


def type_of_rest_basis(data):
    return fem.TetP1Basis if isinstance(data, mesh.TetMesh) else fem.HexQ1Basis


def test_single_tet_energy_matches_material_times_volume():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
    data = mesh.TetMesh(nodes, np.array([[0, 1, 2, 3]], dtype=np.int32))
    elastic = volumetric_energy(data, fem.TetP1Basis, fem.TetCentroidQuadrature, 3)
    A = np.eye(3) + 0.1 * np.array([[1, 2, 0], [0, 1, 3], [1, 0, 2.0]])
    u = (nodes @ (A - np.eye(3)).T).ravel()
    expected = fem.NeoHookean.eval_W(A, np.eye(3), NEO_HOOKEAN_THETA) / 6
    assert elastic.eval_value(u) == pytest.approx(expected, rel=1e-13)
    assert isinstance(elastic, fem.VolumetricFemEnergy) and isinstance(elastic, energy.Energy)


def test_erased_field_types_can_be_mixed():
    data = mesh.read_veg(ASSETS / "box-hex.veg")
    n = data.topology.num_cells
    Q1 = fem.InterpolatedField3D[fem.HexQ1Basis]
    # theta interpolated from Q1 nodal values equals the constant field everywhere.
    interpolated_theta = [Q1(np.tile(NEO_HOOKEAN_THETA, (8, 1)))] * n
    constant = volumetric_energy(data, fem.HexHermiteBasis, fem.GaussLegendre3D[2], 24)
    mixed = volumetric_energy(data, fem.HexHermiteBasis, fem.GaussLegendre3D[2], 24, interpolated_theta)
    u = 1e-3 * np.sin(np.arange(constant.num_dofs))
    assert mixed.eval_value(u) == pytest.approx(constant.eval_value(u), rel=1e-14)
    np.testing.assert_allclose(mixed.eval_gradient(u), constant.eval_gradient(u), rtol=1e-12, atol=1e-14)


def test_energy_rejects_bad_fields():
    data = mesh.read_veg(ASSETS / "box-tet.veg")
    Energy = fem.VolumetricFemEnergy[fem.NeoHookean, fem.TetP1Basis, fem.TetCentroidQuadrature]
    rest_map = dof.VertexDofMap(data.topology, 3)
    X = rest_fields(fem.InterpolatedField3D[fem.TetP1Basis], rest_map, data.positions.ravel())
    n = data.topology.num_cells
    with pytest.raises(ValueError):
        Energy(rest_map, X[:-1], constant_fields(np.eye(3), n), constant_fields(NEO_HOOKEAN_THETA, n))
    with pytest.raises(ValueError):
        Energy(rest_map, X, constant_fields(np.eye(2), n), constant_fields(NEO_HOOKEAN_THETA, n))
    with pytest.raises(TypeError):
        Energy(rest_map, X, [np.eye(3)] * n, constant_fields(NEO_HOOKEAN_THETA, n))


# examples/fem/**/*.py mirror the C++ examples line by line. They solve with AutoSparseSolver, whose
# threaded factorizations (PARDISO, Accelerate) need not repeat bit for bit, so the mirrors agree
# to rounding: exact counts and connectivity, values to their printed digits, positions to 1e-10.


def parse_equilibrium(stdout):
    values = dict(item.split("=") for item in stdout.strip().split(", "))
    return {key: int(value) if key in ("converged", "iterations") else float(value) for key, value in values.items()}


@pytest.mark.parametrize(
    "name,suffix",
    [
        ("tet_p1_corner_pull", ".veg"),
        ("hex_q1_corner_pull", ".veg"),
        ("hex_hermite_corner_pull", ".veg"),
        ("shell_saddle_pull", ".obj"),
    ],
)
def test_python_examples_match_cpp(tmp_path, name, suffix):
    executable = EXAMPLES / "equilibrium" / name
    if not executable.exists():
        pytest.skip(f"{executable} is not built")
    script = Path(__file__).resolve().parents[2] / "examples" / "fem" / "equilibrium" / f"{name}.py"
    asset = ASSETS / {"tet_p1_corner_pull": "box-tet.veg", "shell_saddle_pull": "shell.obj"}.get(name, "box-hex.veg")
    cpp_out, py_out = tmp_path / f"cpp{suffix}", tmp_path / f"py{suffix}"
    cpp = subprocess.run([executable, asset, cpp_out], check=True, capture_output=True, text=True)
    py = subprocess.run([sys.executable, script, asset, py_out], check=True, capture_output=True, text=True)
    cpp_results, py_results = parse_equilibrium(cpp.stdout), parse_equilibrium(py.stdout)
    assert py_results.keys() == cpp_results.keys()
    for key, value in cpp_results.items():
        if key in ("converged", "iterations"):
            assert py_results[key] == value, key
        elif key == "free residual":  # rounding noise below the 1e-8 tolerance
            assert value < 1e-8 and py_results[key] < 1e-8
        else:
            assert py_results[key] == pytest.approx(value, rel=1e-5), key
    outputs = [(mesh.read_veg, suffix)] if suffix == ".veg" else []
    outputs.append((mesh.read_obj, ".obj"))  # volume examples also write the boundary surface
    for read, extension in outputs:
        cpp_mesh, py_mesh = read(cpp_out.with_suffix(extension)), read(py_out.with_suffix(extension))
        np.testing.assert_array_equal(py_mesh.cells, cpp_mesh.cells)
        np.testing.assert_allclose(py_mesh.positions, cpp_mesh.positions, rtol=0, atol=1e-10)


# examples/fem/plasticity: the plastic displacement u_p twists the bar by 90 degrees and Fp is
# DeformationGradientField3D(u_p, X). u_p is interpolated in the displacement space (C1 Hermite for the
# Hermite example), so the released bar takes the shape X + u_p stress-free. Clamped, it keeps residual
# stress.


def parse_plastic_twist(stdout):
    results = {}
    for line in stdout.splitlines():
        case, fields = line.split(": ", 1)
        values = dict(item.split("=") for item in fields.split(", "))
        results[case] = {
            "converged": values["converged"] == "1",
            "iterations": int(values["iterations"]),
            "elastic": float(values["elastic"]),
            "twist": float(values["top twist"].removesuffix(" deg")),
        }
    return results


@pytest.mark.parametrize(
    "name,asset",
    [
        ("tet_p1_plastic_twist", "box-tet.veg"),
        ("hex_q1_plastic_twist", "box-hex.veg"),
        ("hex_hermite_plastic_twist", "box-hex.veg"),
    ],
)
def test_plastic_twist_examples(tmp_path, name, asset):
    executable = EXAMPLES / "plasticity" / name
    if not executable.exists():
        pytest.skip(f"{executable} is not built")
    script = Path(__file__).resolve().parents[2] / "examples" / "fem" / "plasticity" / f"{name}.py"
    cpp = subprocess.run([executable, ASSETS / asset, tmp_path / "cpp.veg"], check=True, capture_output=True, text=True)
    py = subprocess.run(
        [sys.executable, script, ASSETS / asset, tmp_path / "py.veg"], check=True, capture_output=True, text=True
    )
    # The C++ compiler may contract the examples' u_p arithmetic into fused multiply-adds, so the
    # mirrors agree to rounding rather than bit for bit.
    cpp_results, py_results = parse_plastic_twist(cpp.stdout), parse_plastic_twist(py.stdout)
    for case in ("free", "clamped"):
        assert py_results[case]["converged"] == cpp_results[case]["converged"]
        assert py_results[case]["iterations"] == cpp_results[case]["iterations"]
        assert py_results[case]["elastic"] == pytest.approx(cpp_results[case]["elastic"], rel=1e-9, abs=1e-12)
        assert py_results[case]["twist"] == pytest.approx(cpp_results[case]["twist"], abs=1e-9)
        cpp_mesh, py_mesh = mesh.read_veg(tmp_path / f"cpp_{case}.veg"), mesh.read_veg(tmp_path / f"py_{case}.veg")
        np.testing.assert_array_equal(py_mesh.cells, cpp_mesh.cells)
        np.testing.assert_allclose(py_mesh.positions, cpp_mesh.positions, rtol=0, atol=1e-12)
        cpp_surface = mesh.read_obj(tmp_path / f"cpp_{case}.obj")
        py_surface = mesh.read_obj(tmp_path / f"py_{case}.obj")
        np.testing.assert_array_equal(py_surface.cells, cpp_surface.cells)
        np.testing.assert_allclose(py_surface.positions, cpp_surface.positions, rtol=0, atol=1e-12)
        if name == "hex_hermite_plastic_twist":
            # Four samples per edge produce 16 triangles per original boundary triangle.
            coarse = mesh.extract_surface(cpp_mesh)
            assert len(cpp_surface.cells) == 16 * len(coarse.cells)
            assert len(cpp_surface.positions) > len(coarse.positions)
            assert np.isfinite(cpp_surface.positions).all()

    if name == "tet_p1_plastic_twist":
        for run, prefix in ((cpp, "cpp"), (py, "py")):
            for line in run.stdout.splitlines():
                fields = dict(item.split("=") for item in line.split(": ", 1)[1].split(", "))
                assert float(fields["free residual"]) <= 1e-8
                assert float(fields["max displacement"]) > 0
            rest_surface = mesh.read_obj(tmp_path / f"{prefix}_rest.obj")
            final_surface = mesh.read_obj(tmp_path / f"{prefix}_free.obj")
            np.testing.assert_array_equal(rest_surface.cells, final_surface.cells)
            assert np.isfinite(final_surface.positions).all()
            original = mesh.read_veg(ASSETS / asset)
            final = mesh.read_veg(tmp_path / f"{prefix}_free.veg")
            bottom = np.isclose(original.positions[:, 2], original.positions[:, 2].min())
            np.testing.assert_allclose(final.positions[bottom], original.positions[bottom], atol=1e-14)
    free, clamped = cpp_results["free"], cpp_results["clamped"]
    assert free["converged"] and clamped["converged"]
    assert free["twist"] == pytest.approx(90.0, abs=1e-6)
    assert free["elastic"] < 1e-12
    assert clamped["twist"] == pytest.approx(0.0, abs=1e-9)
    assert clamped["elastic"] > 0.05  # held in place, the bar keeps residual stress


# Building blocks: one element assembled by hand must agree with the mesh energies.


def single_hex(scale=1.5):
    nodes = scale * np.array([fem.HexQ1Basis.reference_node(v) for v in range(8)])
    return mesh.HexMesh(nodes, np.arange(8, dtype=np.int32)[None])


def hex_local_energy(Local, data, u):
    X_field = fem.InterpolatedField3D[fem.HexQ1Basis](data.positions)
    u_field = fem.InterpolatedField3D[fem.HexQ1Basis](u.reshape(8, 3))
    F = fem.DeformationGradientField3D(u_field, X_field)
    Fp = fem.IdentityField3D(np.eye(3).ravel())
    theta = fem.IdentityField3D(NEO_HOOKEAN_THETA.copy())
    return X_field, Local[fem.NeoHookean](F, Fp, theta)


def test_element_matches_mesh_energy():
    data = single_hex()
    u = 0.05 * np.sin(np.arange(24.0))
    X_field, local = hex_local_energy(fem.VolumetricMaterialLocalEnergyF3D, data, u)
    assert isinstance(local, fem.LocalEnergy3D) and isinstance(local, fem.VolumetricMaterialLocalEnergyF3D)
    element = fem.VolumetricElement3D[fem.GaussLegendre3D[2]](X_field)
    value, gradient, hessian = element.eval(local, gradient=True, hessian=True)
    assert element.eval(local)[1:] == (None, None)
    whole = volumetric_energy(data, fem.HexQ1Basis, fem.GaussLegendre3D[2], 3)
    assert value == pytest.approx(whole.eval_value(u), rel=1e-14)
    np.testing.assert_allclose(gradient, whole.eval_gradient(u), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(hessian, whole.eval_hessian(u).toarray(), rtol=1e-12, atol=1e-13)
    collapse = -2 * data.positions.ravel()
    assert 0 < element.max_feasible_step(local, collapse) < 0.5
    assert element.max_feasible_step(local, np.zeros(24)) == 1.0


def test_shell_local_energy_matches_shell_energy():
    X = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.0]])
    topology = mesh.TriangleTopology(3, np.array([[0, 1, 2]], dtype=np.int32))
    stencils = mesh.make_triangle_mid_edge_stencils(topology)
    stencil_map = dof.StencilDofMap(3, stencils.vertices, 3)
    rest = fem.TriangleMidEdgeRestData(X, stencils.opposite[0])
    theta = np.array([1.0, 2.0])
    whole = fem.ShellFemEnergy[fem.KoiterStVK](stencil_map, [rest], [0.1], [theta])
    u = 0.1 * np.cos(np.arange(9.0))
    local = fem.ShellMaterialLocalEnergy2D[fem.KoiterStVK](rest, u.reshape(3, 3), 0.1, theta)
    area = 0.5 * rest.sqrt_det_a_bar
    value, gradient, _ = local.accumulate(np.full(2, 1 / 3), area, gradient=True)
    assert value == pytest.approx(whole.eval_value(u), rel=1e-14)
    np.testing.assert_allclose(gradient, whole.eval_gradient(u), rtol=1e-12, atol=1e-14)


def test_triangle_mid_edge_kinematics():
    x = np.array([[0, 0, 0], [2, 0, 0.1], [0, 1, 0.2]])
    opposite = np.array([-1, -1, -1], dtype=np.int32)
    kinematics, dab_dx, d2ab_dx2 = fem.eval_triangle_mid_edge_kinematics(x, opposite, gradient=True, hessian=True)
    J = np.column_stack([x[1] - x[0], x[2] - x[0]])
    np.testing.assert_allclose(kinematics.a, J.T @ J, atol=1e-14)
    assert dab_dx.shape == (6, 9) and len(d2ab_dx2) == 6
    h = 1e-6
    for k in range(9):
        step = np.zeros(9)
        step[k] = h
        plus = fem.eval_triangle_mid_edge_kinematics(x + step.reshape(3, 3), opposite)[0].a
        minus = fem.eval_triangle_mid_edge_kinematics(x - step.reshape(3, 3), opposite)[0].a
        fd = (plus - minus) / (2 * h)
        np.testing.assert_allclose(dab_dx[:3, k], [fd[0, 0], fd[0, 1], fd[1, 1]], atol=1e-8)
    assert fem.eval_triangle_mid_edge_feasible_step(x, -x, opposite) < 1


def test_kinematic_helpers():
    assert fem.eval_positive_det_feasible_step(np.eye(3), -2 * np.eye(3)) < 0.5
    assert fem.eval_positive_det_feasible_step(np.eye(3), np.eye(3)) == 1.0
    rng = np.random.default_rng(1)
    D, g, H = rng.standard_normal((9, 5)), rng.standard_normal((3, 3)), rng.standard_normal((9, 9))
    np.testing.assert_allclose(fem.eval_gradient_wrt_dofs(D, g), D.T @ g.ravel(order="F"))
    np.testing.assert_allclose(fem.eval_hessian_wrt_dofs(D, H), D.T @ H @ D, atol=1e-12)


def test_basis_dof_transform_maps_physical_hermite_dofs():
    size, origin = np.array([2.0, 1.0, 3.0]), np.array([0.5, -1.0, 2.0])
    X_nodes = origin + size * np.array([fem.HexQ1Basis.reference_node(v) for v in range(8)])
    X_field = fem.InterpolatedField3D[fem.HexQ1Basis](X_nodes)
    transform = fem.BasisDofTransform[fem.HexHermiteBasis](X_field)
    assert transform.dofs_per_vertex == 24
    # An affine displacement u = A X: physical slots u, u_X, u_Y, u_Z, then zero mixed derivatives.
    A = np.array([[0.1, -0.2, 0.05], [0.3, 0.0, -0.1], [0.02, 0.15, 0.2]])
    local = np.hstack([X_nodes @ A.T, np.tile(A.T.ravel(), (8, 1)), np.zeros((8, 12))])
    u_field = fem.InterpolatedField3D[fem.HexHermiteBasis](transform.to_coefficients(local))
    F = fem.DeformationGradientField3D(u_field, X_field)
    for xi in ([0.2, 0.7, 0.4], [0.9, 0.1, 0.5]):
        np.testing.assert_allclose(F.eval_value(np.array(xi)), (np.eye(3) + A).ravel(order="F"), atol=1e-13)
    # Pullback scales each coefficient by its slot's element lengths: 1, hx, hy, hz, hx hy, hx hz, hy hz, hx hy hz.
    gradient, hessian = transform.pullback(gradient=np.ones(192))
    assert hessian is None
    hx, hy, hz = size
    slots = np.array([1, hx, hy, hz, hx * hy, hx * hz, hy * hz, hx * hy * hz])
    np.testing.assert_allclose(gradient, np.tile(np.repeat(slots, 3), 8))
    nodal = fem.BasisDofTransform[fem.HexQ1Basis](X_field)
    assert nodal.dofs_per_vertex == 3
    np.testing.assert_array_equal(nodal.to_coefficients(X_nodes), X_nodes)


def test_volume_equilibrium_material_fit_python_matches_cpp():
    import re

    script = ASSETS.parent / "examples/fem/fitting/volume_equilibrium_material_fit.py"
    outputs = []
    for command in ([str(EXAMPLES / "fitting" / "volume_equilibrium_material_fit")], [sys.executable, str(script)]):
        completed = subprocess.run(command, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        outputs.append(completed.stdout)
    values = [np.array([float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", output)])
              for output in outputs]
    assert values[0].size == values[1].size == 8
    np.testing.assert_allclose(values[0], values[1], rtol=1e-3, atol=1e-9)


def test_neo_hookean_mixed_parameter_derivative():
    F = np.array([[1.2, .3, -.1], [.1, .9, .2], [-.05, .1, 1.1]])
    Fp = np.array([[1.3, .2, .1], [0., .9, -.1], [.1, .2, 1.2]])
    theta = np.array([2., 3.])  # lambda, mu
    B = fem.NeoHookean.eval_d2W_dF_dtheta(F, Fp, theta)
    assert B.shape == (9, 2)
    inv_p = np.linalg.inv(Fp)
    inv_F_T = np.linalg.inv(F).T
    expected = np.column_stack([
        (np.linalg.det(Fp) * np.log(np.linalg.det(F @ inv_p)) * inv_F_T).ravel(order="F"),
        (np.linalg.det(Fp) * (F @ inv_p @ inv_p.T - inv_F_T)).ravel(order="F"),
    ])
    np.testing.assert_allclose(B, expected, rtol=1e-12, atol=1e-14)
    for parameters in (np.array([0., 1.]), np.array([5., .7]), theta):
        np.testing.assert_allclose(B @ parameters,
                                  fem.NeoHookean.eval_dW_dF(F, Fp, parameters).ravel(order="F"),
                                  rtol=1e-12, atol=1e-14)
    with pytest.raises(ValueError):
        fem.NeoHookean.eval_d2W_dF_dtheta(F, Fp, np.ones(1))
    with pytest.raises(ValueError):
        fem.NeoHookean.eval_d2W_dF_dtheta(F, Fp, np.array([2., 0.]))
    with pytest.raises(ValueError):
        fem.NeoHookean.eval_d2W_dF_dtheta(-np.eye(3), np.eye(3), theta)
