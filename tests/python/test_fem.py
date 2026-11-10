from litepgo.energy import AffineResponse
import gc
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from litepgo import _core, dof, energy, fem, mesh

ASSETS = Path(__file__).resolve().parents[2] / "assets"
HERMITE_TEST_MESH = Path(__file__).resolve().parents[1] / "assets" / "box-hex.veg"
# The extension lives in build/<preset>/python/litepgo; the examples in build/<preset>/examples/fem.
EXAMPLES = Path(_core.__file__).parents[2] / "examples" / "fem"

NEO_HOOKEAN_THETA = np.array([2.0, 3.0])


def test_templates_mirror_cpp_arguments():
    Energy = fem.VolumetricFemEnergy[fem.HexQ1Basis, fem.GaussLegendre3D[2]]
    assert Energy.__name__ == "VolumetricFemEnergy[HexQ1Basis, GaussLegendre3D[2]]"
    assert issubclass(Energy, energy.Energy) and issubclass(Energy, fem.VolumetricFemEnergy)
    assert fem.InterpolatedResponse[fem.TetP1Basis] is fem.InterpolatedResponse[(fem.TetP1Basis,)]
    assert issubclass(fem.InterpolatedResponse[fem.TetP1Basis], energy.Response)
    assert fem.GaussLegendre3D[4].num_points == 64
    with pytest.raises(TypeError, match=r"available: .*GaussLegendre3D\[2\]"):
        fem.GaussLegendre3D[3]
    with pytest.raises(TypeError, match="VolumetricFemEnergy"):
        fem.VolumetricFemEnergy[fem.TetP1Basis, fem.GaussLegendre3D[2]]
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


def test_basis_coefficient_points_interpolate():
    for i in range(fem.TetP1Basis.num_basis_functions):
        np.testing.assert_allclose(fem.TetP1Basis.eval_N(fem.TetP1Basis.coefficient_info(i).point), np.eye(4)[i], atol=1e-15)


@pytest.mark.parametrize(
    "basis,dim,hermite",
    [
        (fem.LinearLagrangeBasis, 1, False),
        (fem.CubicHermiteBasis, 1, True),
        (fem.TriangleP1Basis, 2, False),
        (fem.TetP1Basis, 3, False),
        (fem.HexQ1Basis, 3, False),
        (fem.QuadHermiteBasis, 2, True),
        (fem.HexHermiteBasis, 3, True),
    ],
)
def test_coefficient_info_samples_reconstruct_polynomials(basis, dim, hermite):
    from itertools import product
    from math import factorial

    assert basis.__name__ in fem.__all__
    assert not hasattr(basis, "reference_node")
    assert f"CoefficientInfo{dim}D" in fem.__all__
    samples = [basis.coefficient_info(i) for i in range(basis.num_basis_functions)]
    for i, info in enumerate(samples):
        assert isinstance(info, getattr(fem, f"CoefficientInfo{dim}D"))
        assert isinstance(info.derivative_order, tuple)
        assert len(info.derivative_order) == dim
        assert np.asarray(info.point).shape == (() if dim == 1 else (dim,))
        with pytest.raises(AttributeError):
            info.derivative_order = (0,) * dim
        with pytest.raises(AttributeError):
            info.point = 0.0
    with pytest.raises(IndexError):
        basis.coefficient_info(-1)
    with pytest.raises(IndexError):
        basis.coefficient_info(basis.num_basis_functions)

    def monomial(point, powers, derivative):
        value = 1.0
        for x, p, d in zip(np.atleast_1d(point), powers, derivative):
            if p < d:
                return 0.0
            value *= factorial(p) / factorial(p - d) * x ** (p - d)
        return value

    if basis in (fem.TriangleP1Basis, fem.TetP1Basis):
        powers = [(0,) * dim] + [tuple(row) for row in np.eye(dim, dtype=int)]
    else:
        powers = list(product(range(4 if hermite else 2), repeat=dim))
    xi = 0.37 if dim == 1 else np.linspace(0.2, 0.7, dim)
    first = np.eye(dim, dtype=int)
    second = [first[i] + first[j] for i in range(dim) for j in range(i, dim)]
    for power in powers:
        coefficients = np.array([
            monomial(info.point, power, info.derivative_order) for info in samples
        ])
        assert coefficients @ basis.eval_N(xi) == pytest.approx(
            monomial(xi, power, (0,) * dim), abs=1e-12
        )
        np.testing.assert_allclose(
            coefficients @ basis.eval_dN_dxi(xi),
            [monomial(xi, power, order) for order in first],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            coefficients @ basis.eval_d2N_dxi2(xi),
            [monomial(xi, power, order) for order in second],
            atol=1e-12,
        )


@pytest.mark.parametrize(
    "quadrature,dim,volume",
    [
        (fem.TetCentroidQuadrature, 3, 1 / 6),
        (fem.TriangleCentroidQuadrature, 2, 1 / 2),
        (fem.GaussLegendre1D[2], 1, 1.0),
        (fem.GaussLegendre1D[4], 1, 1.0),
        (fem.GaussLegendre2D[2], 2, 1.0),
        (fem.GaussLegendre2D[4], 2, 1.0),
        (fem.GaussLegendre3D[2], 3, 1.0),
        (fem.GaussLegendre3D[4], 3, 1.0),
    ],
)
def test_quadrature_weights(quadrature, dim, volume):
    points, weights = quadrature.points(), quadrature.weights()
    assert points.shape == (quadrature.num_points, dim)
    assert weights.shape == (quadrature.num_points,)
    assert weights.sum() == pytest.approx(volume)
    expected_points = points.copy()
    points[:] = -1
    weights[:] = 0
    np.testing.assert_array_equal(quadrature.points(), expected_points)
    assert quadrature.weights().sum() == pytest.approx(volume)


@pytest.mark.parametrize("num_points", [2, 4])
def test_gauss_legendre_1d_integrates_polynomials(num_points):
    assert "GaussLegendre1D" in fem.__all__
    Q = fem.GaussLegendre1D[num_points]
    assert Q is fem.GaussLegendre1D[(num_points,)]
    assert Q.num_points == num_points
    points, weights = Q.points()[:, 0], Q.weights()
    assert np.all((points > 0) & (points < 1))
    assert np.all(np.diff(points) > 0)
    assert np.all(weights > 0)
    for degree in range(2 * num_points):
        assert weights @ points**degree == pytest.approx(1 / (degree + 1), rel=0, abs=1e-14)
    with pytest.raises(TypeError, match=r"available: .*GaussLegendre1D\[2\]"):
        fem.GaussLegendre1D[3]


def test_fields():
    coeffs = np.array([fem.HexQ1Basis.coefficient_info(v).point for v in range(8)]) * [2.0, 1.0, 3.0]
    X = fem.InterpolatedResponse[fem.HexQ1Basis](coeffs)
    xi = np.array([0.2, 0.5, 0.9])
    np.testing.assert_allclose(X.evaluate(xi).value, [0.4, 0.5, 2.7])
    np.testing.assert_allclose(X.evaluate(xi).jacobian().toarray(), np.diag([2.0, 1.0, 3.0]), atol=1e-15)
    assert (X.num_outputs, X.num_inputs) == (3, 3)
    constant = AffineResponse(np.zeros((np.asarray(np.array([1.0, 2.0])).size, 3)), np.array([1.0, 2.0]))
    np.testing.assert_array_equal(constant.evaluate(xi).value, [1.0, 2.0])
    with pytest.raises(ValueError):
        fem.InterpolatedResponse[fem.HexQ1Basis](coeffs[:7])


def test_deformation_gradient_keeps_its_fields_alive():
    def make():
        nodes = np.array([fem.TetP1Basis.coefficient_info(v).point for v in range(4)])
        u = fem.InterpolatedResponse[fem.TetP1Basis](0.1 * nodes)
        X = fem.InterpolatedResponse[fem.TetP1Basis](nodes)
        return fem.DeformationGradientResponse(u, X)

    F = make()
    gc.collect()
    np.testing.assert_allclose(F.evaluate(np.full(3, 0.25)).value, (1.1 * np.eye(3)).ravel(order='F'))


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
    return [AffineResponse(np.zeros((np.asarray(np.asarray(value, dtype=float).ravel()).size, 3)), np.asarray(value, dtype=float).ravel())] * count


def volumetric_energy(data, basis, quadrature, dofs_per_vertex, theta_fields=None):
    rest_map = dof.VertexDofMap(data.topology, 3)
    X_fields = rest_fields(fem.InterpolatedResponse[type_of_rest_basis(data)], rest_map, data.positions.ravel())
    n = data.topology.num_cells
    Energy = fem.VolumetricFemEnergy[basis, quadrature]
    return Energy(fem.PointwiseVolumetricAdapter[fem.NeoHookean](),
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
    assert elastic.evaluate(u).value == pytest.approx(expected, rel=1e-13)
    assert isinstance(elastic, fem.VolumetricFemEnergy) and isinstance(elastic, energy.Energy)


def test_erased_field_types_can_be_mixed():
    data = mesh.read_veg(ASSETS / "box-hex.veg")
    n = data.topology.num_cells
    Q1 = fem.InterpolatedResponse[fem.HexQ1Basis]
    # theta interpolated from Q1 nodal values equals the constant field everywhere.
    interpolated_theta = [Q1(np.tile(NEO_HOOKEAN_THETA, (8, 1)))] * n
    constant = volumetric_energy(data, fem.HexHermiteBasis, fem.GaussLegendre3D[2], 24)
    mixed = volumetric_energy(data, fem.HexHermiteBasis, fem.GaussLegendre3D[2], 24, interpolated_theta)
    u = 1e-3 * np.sin(np.arange(constant.num_dofs))
    assert mixed.evaluate(u).value == pytest.approx(constant.evaluate(u).value, rel=1e-14)
    np.testing.assert_allclose(mixed.evaluate(u).gradient(), constant.evaluate(u).gradient(), rtol=1e-12, atol=1e-14)


def test_energy_rejects_bad_fields():
    data = mesh.read_veg(ASSETS / "box-tet.veg")
    Energy = fem.VolumetricFemEnergy[fem.TetP1Basis, fem.TetCentroidQuadrature]
    rest_map = dof.VertexDofMap(data.topology, 3)
    X = rest_fields(fem.InterpolatedResponse[fem.TetP1Basis], rest_map, data.positions.ravel())
    n = data.topology.num_cells
    with pytest.raises(ValueError):
        Energy(fem.PointwiseVolumetricAdapter[fem.NeoHookean](), rest_map, X[:-1], constant_fields(np.eye(3), n), constant_fields(NEO_HOOKEAN_THETA, n))
    with pytest.raises(ValueError):
        Energy(fem.PointwiseVolumetricAdapter[fem.NeoHookean](), rest_map, X, constant_fields(np.eye(2), n), constant_fields(NEO_HOOKEAN_THETA, n))
    with pytest.raises(TypeError):
        Energy(fem.PointwiseVolumetricAdapter[fem.NeoHookean](), rest_map, X, [np.eye(3)] * n, constant_fields(NEO_HOOKEAN_THETA, n))


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
    if sys.platform == "win32":
        executable = executable.with_suffix(".exe")
    if not executable.exists():
        pytest.skip(f"{executable} is not built")
    script = Path(__file__).resolve().parents[2] / "examples" / "fem" / "equilibrium" / f"{name}.py"
    asset = ASSETS / {"tet_p1_corner_pull": "box-tet.veg", "shell_saddle_pull": "shell.obj"}.get(name, "box-hex.veg")
    if name == "hex_hermite_corner_pull":
        asset = HERMITE_TEST_MESH
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
# DeformationGradientResponse(u_p, X). u_p is interpolated in the displacement space (C1 Hermite for the
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
    if sys.platform == "win32":
        executable = executable.with_suffix(".exe")
    if not executable.exists():
        pytest.skip(f"{executable} is not built")
    script = Path(__file__).resolve().parents[2] / "examples" / "fem" / "plasticity" / f"{name}.py"
    input_mesh = ASSETS / asset
    if name == "hex_hermite_plastic_twist":
        input_mesh = HERMITE_TEST_MESH
    cpp = subprocess.run([executable, input_mesh, tmp_path / "cpp.veg"], check=True, capture_output=True, text=True)
    py = subprocess.run(
        [sys.executable, script, input_mesh, tmp_path / "py.veg"], check=True, capture_output=True, text=True
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
    nodes = scale * np.array([fem.HexQ1Basis.coefficient_info(v).point for v in range(8)])
    return mesh.HexMesh(nodes, np.arange(8, dtype=np.int32)[None])






def test_shell_material_integration_matches_shell_energy():
    X = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.0]])
    topology = mesh.TriangleTopology(3, np.array([[0, 1, 2]], dtype=np.int32))
    stencils = mesh.make_triangle_mid_edge_stencils(topology)
    stencil_map = dof.StencilDofMap(3, stencils.vertices, 3)
    rest = fem.TriangleMidEdgeRestData(X, stencils.opposite[0])
    theta = np.array([1.0, 2.0])
    whole = fem.ShellFemEnergy(fem.PointwiseShellAdapter[fem.KoiterStVK](), stencil_map, [rest], [0.1], [theta])
    u = 0.1 * np.cos(np.arange(9.0))
    current, J, second = fem.eval_triangle_mid_edge_kinematics(
        X + u.reshape(3, 3), rest.opposite, gradient=True, hessian=True
    )
    inputs = (current.a, current.b, rest.a_bar, rest.b_bar, 0.1, theta)
    area = 0.5 * rest.sqrt_det_a_bar
    stress = fem.KoiterStVK.eval_dW_dab(*inputs)
    tangent = fem.KoiterStVK.eval_d2W_dab2(*inputs)
    value = area * fem.KoiterStVK.eval_W(*inputs)
    gradient = area * J.T @ stress
    hessian = area * (J.T @ tangent @ J + sum(stress[i] * second[i] for i in range(6)))
    assert value == pytest.approx(whole.evaluate(u).value, rel=1e-14)
    np.testing.assert_allclose(gradient, whole.evaluate(u).gradient(), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(hessian, whole.evaluate(u).hessian().toarray(), rtol=1e-12, atol=1e-14)


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
    X_nodes = origin + size * np.array([fem.HexQ1Basis.coefficient_info(v).point for v in range(8)])
    X_field = fem.InterpolatedResponse[fem.HexQ1Basis](X_nodes)
    transform = fem.BasisDofTransform[fem.HexHermiteBasis](X_field)
    assert transform.dofs_per_vertex == 24
    # An affine displacement u = A X: physical slots u, u_X, u_Y, u_Z, then zero mixed derivatives.
    A = np.array([[0.1, -0.2, 0.05], [0.3, 0.0, -0.1], [0.02, 0.15, 0.2]])
    local = np.hstack([X_nodes @ A.T, np.tile(A.T.ravel(), (8, 1)), np.zeros((8, 12))])
    u_field = fem.InterpolatedResponse[fem.HexHermiteBasis](transform.to_coefficients(local))
    F = fem.DeformationGradientResponse(u_field, X_field)
    for xi in ([0.2, 0.7, 0.4], [0.9, 0.1, 0.5]):
        np.testing.assert_allclose(F.evaluate(np.array(xi)).value, (np.eye(3) + A).ravel(order='F'), atol=1e-13)
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
