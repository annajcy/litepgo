"""Python materials (PyVolumetricMaterial, PyShellMaterial and Torch subclasses) in native FEM."""

import gc
import weakref

import numpy as np
import pytest
import torch

from litepgo import dof, energy, fem, mesh, solver
from litepgo.energy import AffineResponse
from litepgo.torch import TorchShellMaterial, TorchVolumetricMaterial

KINDS = (energy.HessianKind.Exact, energy.HessianKind.Projected)


def vec_to_tensor(matrix, trailing):
    """Native rows vec(dW/dF) (i+3j); columns vec(.) (k+3l) or a parameter index."""
    out = np.empty((3, 3) + ((3, 3) if trailing == (9,) else trailing))
    for i in range(3):
        for j in range(3):
            row = matrix[i + 3 * j]
            out[i, j] = row.reshape(3, 3, order="F") if trailing == (9,) else row
    return out


class PointwiseNeoHookean(fem.PyVolumetricMaterial):
    """Native NeoHookean called point by point; checks every index conversion."""

    num_parameters = 2
    calls = []

    @staticmethod
    def eval(F, Fp, theta, *, outputs):
        PointwiseNeoHookean.calls.append((len(F), tuple(outputs)))
        N = fem.NeoHookean
        points = list(zip(F, Fp, theta))
        table = {
            "W": lambda f, p, t: N.eval_W(f, p, t),
            "dW_dF": lambda f, p, t: N.eval_dW_dF(f, p, t),
            "d2W_dF2": lambda f, p, t: vec_to_tensor(N.eval_d2W_dF2(f, p, t), (9,)),
            "d2W_dF_dtheta": lambda f, p, t: vec_to_tensor(N.eval_d2W_dF_dtheta(f, p, t), (2,)),
            "d2W_dF_dFp": lambda f, p, t: vec_to_tensor(N.eval_d2W_dF_dFp(f, p, t), (9,)),
        }
        return {name: np.array([table[name](*point) for point in points]) for name in outputs}


class TorchNeoHookean(TorchVolumetricMaterial):
    num_parameters = 2

    @staticmethod
    def W(F, Fp, theta):
        Fe = F @ torch.linalg.inv(Fp)
        log_J = torch.linalg.det(Fe).log()
        lam, mu = theta.unbind(-1)
        psi = 0.5 * mu * (Fe.square().sum((-2, -1)) - 3) - mu * log_J + 0.5 * lam * log_J.square()
        return torch.linalg.det(Fp) * psi


def hexes():
    nodes = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
         [3, 0, 0], [3, 1, 0], [3, 0, 1], [3, 1, 1]], dtype=float)
    cells = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [1, 8, 9, 2, 5, 10, 11, 6]], dtype=np.int32)
    return nodes, mesh.HexTopology(12, cells)


def tets():
    nodes = np.array([[0, 0, 0], [1.2, 0, 0], [0, 1, 0], [0, 0, .8], [1.2, 1, .8]])
    return nodes, mesh.TetTopology(5, np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32))


CASES = [
    (fem.TetP1Basis, fem.TetCentroidQuadrature, tets, fem.TetP1Basis, 3),
    (fem.HexQ1Basis, fem.GaussLegendre3D[2], hexes, fem.HexQ1Basis, 3),
    (fem.HexQ1Basis, fem.GaussLegendre3D[4], hexes, fem.HexQ1Basis, 3),
    (fem.HexHermiteBasis, fem.GaussLegendre3D[2], hexes, fem.HexQ1Basis, 24),
    (fem.HexHermiteBasis, fem.GaussLegendre3D[4], hexes, fem.HexQ1Basis, 24),
]
IDS = ["TetP1", "HexQ1-2", "HexQ1-4", "Hermite-2", "Hermite-4"]


def volume_problem(case):
    basis, quadrature, make_mesh, geometry_basis, dofs_per_vertex = case
    nodes, topology = make_mesh()
    rest_map = dof.VertexDofMap(topology, 3)
    mapping = dof.VertexDofMap(topology, dofs_per_vertex)
    X = [fem.InterpolatedResponse[geometry_basis](rest_map.gather(e, nodes.ravel()))
         for e in range(topology.num_cells)]
    plastic = np.array([[1.08, 0.05, 0.0], [0.02, 0.97, 0.1], [0.0, 0.03, 1.04]])
    Fp = [AffineResponse(np.zeros((9, 3)), (plastic + 0.02 * e).ravel(order="F"))
          for e in range(topology.num_cells)]
    theta = [AffineResponse(np.zeros((2, 3)), np.array([2.0 + e, 3.0 - e])) for e in range(topology.num_cells)]
    u = 0.03 * np.sin(1.3 * np.arange(mapping.num_dofs) + 0.2)
    return topology, mapping, X, Fp, theta, u, (basis, quadrature)


def assert_energy_matches(native, python, u):
    for kind in KINDS:
        a, b = native.eval(u, True, kind), python.eval(u, True, kind)
        assert b[0] == pytest.approx(a[0], rel=1e-12, abs=1e-14)
        np.testing.assert_allclose(b[1], a[1], rtol=1e-11, atol=1e-13)
        np.testing.assert_allclose(b[2].toarray(), a[2].toarray(), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("material", [PointwiseNeoHookean, TorchNeoHookean], ids=["pointwise", "torch"])
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_volumetric_energy_matches_native(case, material):
    topology, mapping, X, Fp, theta, u, key = volume_problem(case)
    native = fem.VolumetricFemEnergy[(fem.NeoHookean, *key)](mapping, X, Fp, theta)
    python = fem.VolumetricFemEnergy[(material, *key)](mapping, X, Fp, theta)
    assert isinstance(python, fem.VolumetricFemEnergy)
    assert isinstance(python, energy.MeshEnergy)
    assert type(python).material is material
    assert_energy_matches(native, python, u)
    assert python.max_feasible_step(u, 30 * u) == pytest.approx(
        fem.VolumetricFemEnergy[(fem.NeoHookean, *key)](mapping, X, Fp, theta).max_feasible_step(u, 30 * u))


@pytest.mark.parametrize("material", [PointwiseNeoHookean, TorchNeoHookean], ids=["pointwise", "torch"])
@pytest.mark.parametrize("case", CASES[:2] + CASES[3:4], ids=["TetP1", "HexQ1-2", "Hermite-2"])
def test_volumetric_force_responses_match_native(case, material):
    topology, mapping, X, Fp, theta, u, key = volume_problem(case)
    count = topology.num_cells
    for Force, fixed, width in ((fem.VolumetricFemForceMaterialResponse, Fp, 2),
                                (fem.VolumetricFemForceFpResponse, theta, 9)):
        for Map, p in ((dof.ConstantDofMap, None), (dof.ElementDofMap, None)):
            parameter_map = Map(topology, width)
            base = np.array([2.5, 1.5]) if width == 2 else np.array([[1.05, .02, 0], [0, .98, .03], [.01, 0, 1.02]]).ravel(order="F")
            p = base if Map is dof.ConstantDofMap else np.concatenate([base * (1 + 0.01 * e) for e in range(count)])
            native = Force[(fem.NeoHookean, *key)](parameter_map, mapping, X, u, fixed)
            python = Force[(material, *key)](parameter_map, mapping, X, u, fixed)
            np.testing.assert_allclose(python.eval_value(p), native.eval_value(p), rtol=1e-11, atol=1e-13)
            np.testing.assert_allclose(python.eval_jacobian(p).toarray(), native.eval_jacobian(p).toarray(),
                                       rtol=1e-10, atol=1e-12)
            v = np.linspace(0.1, 0.4, len(p))
            w = np.linspace(-1, 1, mapping.num_dofs)
            np.testing.assert_allclose(python.eval_jvp(p, v), native.eval_jvp(p, v), rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(python.eval_vjp(p, w), native.eval_vjp(p, w), rtol=1e-10, atol=1e-12)


def test_one_call_per_batch_with_requested_outputs():
    topology, mapping, X, Fp, theta, u, key = volume_problem(CASES[1])
    # Two HexQ1 elements with eight Gauss points each.
    for batch_size, points in ((256, [16]), (1, [8, 8])):
        PointwiseNeoHookean.calls.clear()
        python = fem.VolumetricFemEnergy[(PointwiseNeoHookean, *key)](mapping, X, Fp, theta, batch_size)
        python.eval(u, True, energy.HessianKind.Exact)
        assert PointwiseNeoHookean.calls == [(n, ("W", "dW_dF", "d2W_dF2")) for n in points]
    PointwiseNeoHookean.calls.clear()  # The batch_size=1 energy below calls twice per evaluation.
    python.eval_value(u)
    python.eval_gradient(u)
    response = fem.VolumetricFemForceMaterialResponse[(PointwiseNeoHookean, *key)](
        dof.ConstantDofMap(topology, 2), mapping, X, u, Fp)
    response.eval_value(np.array([2.0, 3.0]))
    response.eval_jacobian(np.array([2.0, 3.0]))
    fp = fem.VolumetricFemForceFpResponse[(PointwiseNeoHookean, *key)](
        dof.ConstantDofMap(topology, 9), mapping, X, u, theta, 256)
    fp.eval_jacobian(np.eye(3).ravel())
    requested = [outputs for _, outputs in PointwiseNeoHookean.calls]
    assert requested == [("W",), ("W",), ("W", "dW_dF"), ("W", "dW_dF"), ("dW_dF",),
                         ("dW_dF", "d2W_dF_dtheta"), ("dW_dF", "d2W_dF_dFp")]


def test_template_dispatch_and_caching():
    key = (fem.TetP1Basis, fem.TetCentroidQuadrature)
    bound = fem.VolumetricFemEnergy[(TorchNeoHookean, *key)]
    assert bound is fem.VolumetricFemEnergy[(TorchNeoHookean, *key)]
    assert bound.__name__ == "VolumetricFemEnergy[TorchNeoHookean, TetP1Basis, TetCentroidQuadrature]"
    assert issubclass(bound, fem.VolumetricFemEnergy)
    assert (fem.PyVolumetricMaterial, *key) in fem.VolumetricFemEnergy.instances
    with pytest.raises(TypeError, match="subclass"):
        fem.VolumetricFemEnergy[(fem.PyVolumetricMaterial, *key)]
    with pytest.raises(TypeError, match="not instantiated"):
        fem.ShellFemEnergy[TorchNeoHookean]
    with pytest.raises(TypeError, match="template indexing"):
        fem.VolumetricFemEnergy.instances[(fem.PyVolumetricMaterial, *key)](
            TorchNeoHookean, *volume_problem(CASES[0])[1:5])


def test_material_class_is_collected_with_its_objects():
    def make():
        class Local(TorchNeoHookean):
            pass

        topology, mapping, X, Fp, theta, u, key = volume_problem(CASES[0])
        potential = fem.VolumetricFemEnergy[(Local, *key)](mapping, X, Fp, theta)
        potential.eval_value(u)
        return weakref.ref(Local), weakref.ref(potential)

    material, potential = make()
    gc.collect()
    assert material() is None and potential() is None


def test_material_errors():
    topology, mapping, X, Fp, theta, u, key = volume_problem(CASES[0])

    def build(**members):
        cls = type("Custom", (fem.PyVolumetricMaterial,), {"num_parameters": 2, **members})
        return fem.VolumetricFemEnergy[(cls, *key)](mapping, X, Fp, theta)

    def evaluator(result):
        return staticmethod(lambda F, Fp, theta, *, outputs: result(len(F)))

    with pytest.raises(TypeError, match="num_parameters"):
        build(num_parameters=None)
    with pytest.raises(TypeError, match="num_parameters"):
        build(num_parameters=True)
    with pytest.raises(ValueError, match="incompatible spatial responses"):
        build(num_parameters=3)
    with pytest.raises(KeyError, match="'W'"):
        build(eval=evaluator(lambda n: {})).eval_value(u)
    with pytest.raises(TypeError, match="dict"):
        build(eval=evaluator(lambda n: np.zeros(n))).eval_value(u)
    with pytest.raises(ValueError, match=r"shape \(3,\); expected \(2,\)"):
        build(eval=evaluator(lambda n: {"W": np.zeros(n + 1)})).eval_value(u)
    with pytest.raises(ValueError, match="dW_dF with shape"):
        build(eval=evaluator(lambda n: {"W": np.zeros(n), "dW_dF": np.zeros((n, 9))})).eval_gradient(u)
    # Non-finite values are domain errors (ValueError); a Newton trial is shortened instead.
    with pytest.raises(ValueError, match="non-finite W"):
        build(eval=evaluator(lambda n: {"W": np.full(n, np.nan)})).eval_value(u)

    def raises(F, Fp, theta, *, outputs):
        raise RuntimeError("from the material")

    with pytest.raises(RuntimeError, match="from the material"):
        build(eval=staticmethod(raises)).eval_value(u)
    with pytest.raises(NotImplementedError):
        fem.VolumetricFemEnergy[(type("Bare", (fem.PyVolumetricMaterial,), {"num_parameters": 2}), *key)](
            mapping, X, Fp, theta).eval_value(u)


def test_feasible_step_hook_and_default():
    topology, mapping, X, Fp, theta, u, key = volume_problem(CASES[0])
    seen = []

    class Halved(TorchNeoHookean):
        @staticmethod
        def eval_feasible_step(F, Fp, theta, dF):
            seen.append((F.shape, dF.shape))
            return np.full(len(F), 0.5)

    hooked = fem.VolumetricFemEnergy[(Halved, *key)](mapping, X, Fp, theta)
    assert hooked.max_feasible_step(u, u) == 0.5
    assert seen == [((2, 3, 3), (2, 3, 3))]
    # Without the hook, the bound is the native NeoHookean det(F) > 0 bound.
    default = fem.VolumetricFemEnergy[(TorchNeoHookean, *key)](mapping, X, Fp, theta)
    native = fem.VolumetricFemEnergy[(fem.NeoHookean, *key)](mapping, X, Fp, theta)
    huge = 100 * np.sin(np.arange(mapping.num_dofs))
    assert 0 < default.max_feasible_step(u, huge) < 1
    assert default.max_feasible_step(u, huge) == pytest.approx(native.max_feasible_step(u, huge))

    class OutOfRange(TorchNeoHookean):
        @staticmethod
        def eval_feasible_step(F, Fp, theta, dF):
            return np.full(len(F), 2.0)

    with pytest.raises(RuntimeError, match="outside"):
        fem.VolumetricFemEnergy[(OutOfRange, *key)](mapping, X, Fp, theta).max_feasible_step(u, u)


def test_newton_rejects_nonfinite_trials_and_matches_native_equilibrium():
    topology, mapping, X, Fp, theta, u, key = volume_problem(CASES[0])

    class Guarded(TorchNeoHookean):
        @staticmethod
        def W(F, Fp, theta):
            W = TorchNeoHookean.W(F, Fp, theta)
            # Reject strongly compressed trials; Newton must back off.
            return torch.where(torch.linalg.det(F) > 0.8, W, torch.full_like(W, float("nan")))

    fixed = dof.DofReduction(15, [*range(9), 12, 13, 14])
    initial = np.zeros(15)
    initial[12:] = [0.04, -0.02, 0.06]
    options = solver.NewtonOptions(gradient_tolerance=1e-10, max_iterations=100)
    native = solver.newton_solve(fem.VolumetricFemEnergy[(fem.NeoHookean, *key)](mapping, X, Fp, theta),
                                 fixed, initial, options)
    python = solver.newton_solve(fem.VolumetricFemEnergy[(Guarded, *key)](mapping, X, Fp, theta),
                                 fixed, initial, options)
    assert python.status == native.status == solver.NewtonStatus.Converged
    np.testing.assert_allclose(python.x, native.x, atol=1e-9)


def test_torch_constant_and_linear_densities_have_zero_derivatives():
    topology, mapping, X, Fp, theta, u, key = volume_problem(CASES[0])

    class Linear(TorchVolumetricMaterial):
        num_parameters = 2

        @staticmethod
        def W(F, Fp, theta):
            return F[:, 0, 0] * 2.0

    python = fem.VolumetricFemEnergy[(Linear, *key)](mapping, X, Fp, theta)
    _, gradient, hessian = python.eval(u, True, energy.HessianKind.Exact)
    assert np.abs(gradient).max() > 0
    assert abs(hessian).max() == 0
    response = fem.VolumetricFemForceMaterialResponse[(Linear, *key)](
        dof.ConstantDofMap(topology, 2), mapping, X, u, Fp)
    assert abs(response.eval_jacobian(np.array([2.0, 3.0]))).max() == 0

    class WrongShape(TorchVolumetricMaterial):
        num_parameters = 2

        @staticmethod
        def W(F, Fp, theta):
            return F.sum()

    with pytest.raises(ValueError, match="float64 tensor of shape"):
        fem.VolumetricFemEnergy[(WrongShape, *key)](mapping, X, Fp, theta).eval_value(u)


def test_torch_material_runs_under_inference_mode():
    topology, mapping, X, Fp, theta, u, key = volume_problem(CASES[0])
    python = fem.VolumetricFemEnergy[(TorchNeoHookean, *key)](mapping, X, Fp, theta)
    native = fem.VolumetricFemEnergy[(fem.NeoHookean, *key)](mapping, X, Fp, theta)
    with torch.inference_mode():
        np.testing.assert_allclose(python.eval_gradient(u), native.eval_gradient(u), rtol=1e-11, atol=1e-13)


# Shell materials


def unpack(packed):
    a = torch.stack([packed[:, 0], packed[:, 1], packed[:, 1], packed[:, 2]], -1).reshape(-1, 2, 2)
    b = torch.stack([packed[:, 3], packed[:, 4], packed[:, 4], packed[:, 5]], -1).reshape(-1, 2, 2)
    return a, b


class TorchKoiter(TorchShellMaterial):
    num_parameters = 2

    @staticmethod
    def W(ab, ab_bar, thickness, theta):
        a, b = unpack(ab)
        a_bar, b_bar = unpack(ab_bar)
        inverse = torch.linalg.inv(a_bar)
        lam, mu = theta.unbind(-1)

        def psi(S):
            M = inverse @ S
            trace = M.diagonal(dim1=-2, dim2=-1).sum(-1)
            return 0.5 * lam * trace.square() + mu * (M @ M).diagonal(dim1=-2, dim2=-1).sum(-1)

        return thickness * psi(0.5 * (a - a_bar)) + thickness**3 / 12 * psi(b - b_bar)


class PointwiseKoiter(fem.PyShellMaterial):
    num_parameters = 2

    @staticmethod
    def eval(ab, ab_bar, thickness, theta, *, outputs):
        def forms(p):
            return np.array([[p[0], p[1]], [p[1], p[2]]]), np.array([[p[3], p[4]], [p[4], p[5]]])

        K = fem.KoiterStVK
        table = {"W": K.eval_W, "dW_dab": K.eval_dW_dab, "d2W_dab2": K.eval_d2W_dab2,
                 "d2W_dab_dtheta": K.eval_d2W_dab_dtheta, "d2W_dab_dabarbbar": K.eval_d2W_dab_dabarbbar}
        result = {name: [] for name in outputs}
        for p, q, t, k in zip(ab, ab_bar, thickness, theta):
            a, b = forms(p)
            a_bar, b_bar = forms(q)
            for name in outputs:
                result[name].append(table[name](a, b, a_bar, b_bar, t, k))
        return {name: np.array(values) for name, values in result.items()}


def shell_problem():
    X = np.array([[0., 0., 0.], [2., 0., 0.], [0., 1., 0.], [2., 2., .1], [-1., 0., -.1], [0., -2., .2]])
    topology = mesh.TriangleTopology(6, np.array([[0, 1, 2], [2, 1, 3], [0, 2, 4], [1, 0, 5]], dtype=np.int32))
    stencil = mesh.make_triangle_mid_edge_stencils(topology)
    mapping = dof.StencilDofMap(6, stencil.vertices, 3)
    rest = [fem.TriangleMidEdgeRestData(mapping.gather(e, X.ravel()), stencil.opposite[e]) for e in range(4)]
    u = .03 * np.linspace(-1, 1, 18)
    u[[11, 14, 17]] += [.2, -.15, .18]
    thickness = [.1, .2, .15, .3]
    theta = [np.array([2., 3.]), np.array([4., 5.]), np.array([-.5, 2.]), np.array([8., 9.])]
    Fp = [np.eye(3) + 0.03 * np.array([[1, .5, 0], [0, -1, .2], [.1, 0, .5]]) * (e + 1) for e in range(4)]
    return topology, mapping, rest, u, thickness, theta, Fp


@pytest.mark.parametrize("material", [PointwiseKoiter, TorchKoiter], ids=["pointwise", "torch"])
def test_shell_energy_and_forces_match_native(material):
    topology, mapping, rest, u, thickness, theta, Fp = shell_problem()
    for batch_size in (1, 256):
        native = fem.ShellFemEnergy[fem.KoiterStVK](mapping, rest, thickness, theta, Fp)
        python = fem.ShellFemEnergy[material](mapping, rest, thickness, theta, Fp, batch_size)
        assert isinstance(python, fem.ShellFemEnergy)
        assert_energy_matches(native, python, u)
        assert python.max_feasible_step(u, 5 * u) == native.max_feasible_step(u, 5 * u)
    for Map in (dof.ConstantDofMap, dof.ElementDofMap):
        p = np.array([2.5, 1.5]) if Map is dof.ConstantDofMap else np.concatenate(theta)
        native = fem.ShellFemForceMaterialResponse[fem.KoiterStVK](Map(topology, 2), mapping, rest, u, thickness)
        python = fem.ShellFemForceMaterialResponse[material](Map(topology, 2), mapping, rest, u, thickness)
        np.testing.assert_allclose(python.eval_value(p), native.eval_value(p), rtol=1e-11, atol=1e-13)
        np.testing.assert_allclose(python.eval_jacobian(p).toarray(), native.eval_jacobian(p).toarray(),
                                   rtol=1e-10, atol=1e-12)
        P = np.array([[1.12, .07, -.02], [.03, .94, .06], [.01, -.03, 1.08]]).ravel(order="F")
        p = P if Map is dof.ConstantDofMap else np.concatenate([P * (1 + .02 * e) for e in range(4)])
        native = fem.ShellFemForceFpResponse[fem.KoiterStVK](Map(topology, 9), mapping, rest, u, thickness, theta)
        python = fem.ShellFemForceFpResponse[material](Map(topology, 9), mapping, rest, u, thickness, theta)
        np.testing.assert_allclose(python.eval_value(p), native.eval_value(p), rtol=1e-11, atol=1e-13)
        np.testing.assert_allclose(python.eval_jacobian(p).toarray(), native.eval_jacobian(p).toarray(),
                                   rtol=1e-10, atol=1e-12)


def test_shell_material_validates_at_construction():
    topology, mapping, rest, u, thickness, theta, Fp = shell_problem()

    class Infinite(PointwiseKoiter):
        @staticmethod
        def eval(ab, ab_bar, thickness, theta, *, outputs):
            return {"W": np.full(len(ab), np.inf)}

    with pytest.raises(ValueError, match="non-finite W"):
        fem.ShellFemEnergy[Infinite](mapping, rest, thickness, theta)
    with pytest.raises(TypeError, match="not instantiated"):
        fem.VolumetricFemEnergy[TorchKoiter, fem.TetP1Basis, fem.TetCentroidQuadrature]
