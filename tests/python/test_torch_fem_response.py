from litepgo.energy import AffineResponse
from pathlib import Path
import runpy

import numpy as np
import pytest
import torch
from litepgo import dof, energy, fem, mesh, solver
from litepgo import torch as bridge


@pytest.fixture
def problem():
    path = (
        Path(__file__).resolve().parents[2] / "examples/fem/fitting/torch_neohookean.py"
    )
    data = runpy.run_path(str(path))["make_problem"]()
    topology = mesh.TetTopology(
        5, np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32)
    )
    return data, topology


def responses(problem, kind, shared, batch_size=1, u=None):
    (nodes, mapping, geometry, plastic, theta, _, E), topology = problem
    if u is None:
        u = (
            nodes
            @ np.array([[0.07, 0.02, 0.01], [0.01, -0.03, 0.0], [0.0, 0.02, 0.05]]).T
        ).ravel()
    width = 2 if kind == "material" else 9
    parameter_map = (dof.ConstantDofMap if shared else dof.ElementDofMap)(
        topology, width
    )
    if kind == "material":
        cls = bridge.TorchVolumetricFemMaterialResponse
        native = fem.VolumetricFemForceMaterialResponse[
            fem.NeoHookean, fem.TetP1Basis, fem.TetCentroidQuadrature
        ]
        fixed = plastic
        blocks = np.array([[2.0, 3.0], [3.0, 2.0]])
    else:
        cls = bridge.TorchVolumetricFemFpResponse
        native = fem.VolumetricFemForceFpResponse[
            fem.NeoHookean, fem.TetP1Basis, fem.TetCentroidQuadrature
        ]
        fixed = theta
        blocks = np.array([f.eval_value(np.full(3, 0.25)) for f in plastic])
    p = blocks[0].copy() if shared else blocks.ravel().copy()
    args = (parameter_map, mapping, geometry, u, fixed)
    return (
        cls(*args, material=E.material, batch_size=batch_size),
        native(*args, batch_size=batch_size),
        p,
        parameter_map,
        u,
    )


@pytest.mark.parametrize("kind", ["material", "fp"])
@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_native_value_jacobian_and_direct_products(problem, kind, shared, batch_size):
    model, native, p, _, u = responses(problem, kind, shared, batch_size)
    value, J = model.eval_value_and_jacobian(p)
    expected, K = native.eval_value_and_jacobian(p)
    np.testing.assert_allclose(value, expected, rtol=1e-12, atol=1e-13)
    (_, mapping, geometry, plastic, theta, _, E), _ = problem
    width = 2 if kind == "material" else 9
    blocks = [p] * 2 if shared else p.reshape(2, width)
    active = [AffineResponse(np.zeros((np.asarray(block).size, 3)), block) for block in blocks]
    potential = bridge.TorchVolumetricFemEnergy(
        mapping,
        geometry,
        plastic if kind == "material" else active,
        active if kind == "material" else theta,
        material=E.material,
        batch_size=batch_size,
    )
    np.testing.assert_allclose(
        value, potential.eval_gradient(u), rtol=1e-12, atol=1e-13
    )
    np.testing.assert_array_equal(J.indptr, K.indptr)
    np.testing.assert_array_equal(J.indices, K.indices)
    np.testing.assert_allclose(J.toarray(), K.toarray(), rtol=1e-11, atol=1e-12)
    rng = np.random.default_rng(15)
    v, w = rng.normal(size=p.size), rng.normal(size=value.size)

    def forbidden(*args):
        raise AssertionError("explicit batch evaluation must not run for products")

    model.eval_batch = forbidden
    jv, jtw = model.eval_jvp(p, v), model.eval_vjp(p, w)
    np.testing.assert_allclose(jv, K @ v, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(jtw, K.T @ w, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(w @ jv, v @ jtw, rtol=1e-12, atol=1e-12)
    h = 1e-6
    np.testing.assert_allclose(
        jv,
        (native.eval_value(p + h * v) - native.eval_value(p - h * v)) / (2 * h),
        rtol=1e-7,
        atol=1e-8,
    )
    with torch.no_grad():
        np.testing.assert_allclose(model.eval_vjp(p, w), jtw)
    with torch.inference_mode():
        np.testing.assert_allclose(model.eval_jvp(p, v), jv)


def factories(problem, kind, use_torch, products_only=False):
    (_, mapping, geometry, plastic, theta, _, example_energy), _ = problem
    response, _, p, parameter_map, _ = responses(problem, kind, True)
    Energy = (
        bridge.TorchVolumetricFemEnergy
        if use_torch
        else fem.VolumetricFemEnergy[
            fem.NeoHookean, fem.TetP1Basis, fem.TetCentroidQuadrature
        ]
    )
    kwargs = {"material": example_energy.material, "batch_size": 1} if use_torch else {}

    def make_energy(p):
        active = [AffineResponse(np.zeros((np.asarray(p).size, 3)), p)] * 2
        return Energy(
            mapping,
            geometry,
            plastic if kind == "material" else active,
            active if kind == "material" else theta,
            **kwargs,
        )

    def make_response(u):
        obj, native, _, _, _ = responses(problem, kind, True, u=u)
        if not use_torch:
            return native
        if products_only:
            evaluate = obj.eval_batch

            def values_only(ids, x, need_jacobian):
                assert not need_jacobian, (
                    "equilibrium products must not request a parameter Jacobian"
                )
                return evaluate(ids, x, False)

            obj.eval_batch = values_only
        return obj

    return make_energy, make_response, p, parameter_map


@pytest.mark.parametrize("kind", ["material", "fp"])
def test_energy_consistency_equilibrium_and_adjoint(problem, kind):
    make_energy, make_response, p, _ = factories(problem, kind, True)
    native_energy, native_response, _, _ = factories(problem, kind, False)
    direct_energy, direct_response, _, _ = factories(
        problem, kind, True, products_only=True
    )
    fixed = dof.DofReduction(15, [*range(9), 12, 13, 14])
    initial = np.zeros(15)
    initial[12:] = [0.04, -0.02, 0.06]
    options = solver.NewtonOptions(gradient_tolerance=1e-12, max_iterations=100)

    def equilibrium(E, R):
        return solver.StaticEquilibriumResponse(len(p), fixed, initial, options, E, R)

    obj, native, direct = (
        equilibrium(make_energy, make_response),
        equilibrium(native_energy, native_response),
        equilibrium(direct_energy, direct_response),
    )
    u, J = obj.eval_value_and_jacobian(p)
    expected_u, expected_J = native.eval_value_and_jacobian(p)
    np.testing.assert_allclose(u, expected_u, atol=1e-11)
    np.testing.assert_allclose(J.toarray(), expected_J.toarray(), atol=1e-10, rtol=1e-9)
    np.testing.assert_array_equal(u[fixed.fixed_dofs], initial[fixed.fixed_dofs])
    np.testing.assert_array_equal(J.toarray()[fixed.fixed_dofs], 0)
    force = make_energy(p).eval_gradient(u)
    np.testing.assert_allclose(make_response(u).eval_value(p), force, atol=1e-12)
    assert np.max(abs(force[fixed.free_dofs])) < 1e-12
    assert np.linalg.norm(J.toarray()[fixed.free_dofs]) > 1e-6
    rng = np.random.default_rng(29)
    v, w = rng.normal(size=p.size), rng.normal(size=15)
    np.testing.assert_allclose(direct.eval_jvp(p, v), J @ v, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(direct.eval_vjp(p, w), J.T @ w, rtol=1e-9, atol=1e-10)
    h = 1e-5
    fd = (obj.eval_value(p + h * v) - obj.eval_value(p - h * v)) / (2 * h)
    np.testing.assert_allclose(direct.eval_jvp(p, v), fd, atol=2e-8, rtol=2e-6)
    x = torch.tensor(p, requires_grad=True)
    y = bridge.response(direct, x)
    (y * torch.tensor(w)).sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), expected_J.T @ w, rtol=1e-9, atol=1e-10)
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(x, torch.tensor(v))
        _, tangent = torch.autograd.forward_ad.unpack_dual(
            bridge.response(direct, dual)
        )
        np.testing.assert_allclose(
            tangent.detach().numpy(), expected_J @ v, rtol=1e-9, atol=1e-10
        )


@pytest.mark.parametrize("kind", ["material", "fp"])
def test_parameter_transforms_keep_products(problem, kind):
    model, _, p, _, _ = responses(problem, kind, True)
    transform = (
        energy.ExpResponse(2) if kind == "material" else fem.LogDiagonalFpResponse(3)
    )
    q = np.log(p) if kind == "material" else np.log([1.04, 0.98, 1.02])
    composed = energy.ComposedResponse(model, transform)
    J = composed.eval_jacobian(q)

    def forbidden(*args):
        raise AssertionError("no explicit Jacobian")

    model.eval_batch = forbidden
    v, w = np.linspace(0.2, 0.4, len(q)), np.linspace(-0.2, 0.1, 15)
    np.testing.assert_allclose(composed.eval_jvp(q, v), J @ v, atol=1e-12)
    np.testing.assert_allclose(composed.eval_vjp(q, w), J.T @ w, atol=1e-12)


@pytest.mark.parametrize("kind", ["material", "fp"])
@pytest.mark.parametrize(
    "material",
    [
        lambda F, P, t: torch.ones(len(F), dtype=torch.float64),
        lambda F, P, t: F.square().sum(dim=(1, 2)),
        lambda F, P, t: t.square().sum(dim=1) + P.square().sum(dim=(1, 2)),
    ],
)
def test_zero_mixed_derivatives(problem, kind, material):
    model, _, p, _, _ = responses(problem, kind, False)
    model.material = material
    assert np.linalg.norm(model.eval_jacobian(p).toarray()) == 0
    assert np.linalg.norm(model.eval_jvp(p, np.ones_like(p))) == 0
    assert np.linalg.norm(model.eval_vjp(p, np.ones(15))) == 0


@pytest.mark.parametrize("kind", ["material", "fp"])
def test_validation_snapshots_and_invalid_trials(problem, kind):
    model, _, p, parameter_map, u = responses(problem, kind, True)
    expected = model.eval_value(p)
    u[:] = 100
    np.testing.assert_array_equal(model.eval_value(p), expected)
    invalid = p.copy()
    invalid[0] = float("nan")
    for operation in (
        lambda: model.eval_value(invalid),
        lambda: model.eval_jvp(invalid, np.ones_like(p)),
        lambda: model.eval_vjp(invalid, np.ones(15)),
    ):
        with pytest.raises(ValueError):
            operation()
    if kind == "fp":
        invalid = -np.eye(3).ravel()
        for operation in (
            lambda: model.eval_value(invalid),
            lambda: model.eval_jvp(invalid, np.ones(9)),
            lambda: model.eval_vjp(invalid, np.ones(15)),
        ):
            with pytest.raises(ValueError):
                operation()
    (_, mapping, geometry, plastic, theta, _, E), topology = problem
    cls = type(model)
    fields = plastic if kind == "material" else theta
    with pytest.raises(ValueError, match="displacement shape"):
        cls(parameter_map, mapping, geometry, np.zeros(14), fields, material=E.material)
    with pytest.raises(TypeError, match="parameter_map"):
        cls(mapping, mapping, geometry, np.zeros(15), fields, material=E.material)
    if kind == "fp":
        with pytest.raises(ValueError, match="nine"):
            cls(
                dof.ConstantDofMap(topology, 8),
                mapping,
                geometry,
                np.zeros(15),
                fields,
                material=E.material,
            )


@pytest.mark.parametrize("kind", ["material", "fp"])
def test_empty_mesh_and_bad_material(problem, kind):
    model, _, p, _, _ = responses(problem, kind, True)
    model.material = lambda F, P, t: F.sum(dim=(1, 2)) * float("nan")
    for evaluate in (
        lambda: model.eval_value(p),
        lambda: model.eval_jvp(p, np.ones_like(p)),
        lambda: model.eval_vjp(p, np.ones(15)),
    ):
        with pytest.raises(ValueError):
            evaluate()
    topology = mesh.TetTopology(0, np.empty((0, 4), dtype=np.int32))
    parameters = dof.ConstantDofMap(topology, len(p))
    mapping = dof.VertexDofMap(topology, 3)
    empty = type(model)(
        parameters, mapping, [], np.zeros(0), [], material=model.material
    )
    assert empty.eval_value(p).size == 0
    assert empty.eval_jacobian(p).shape == (0, len(p))
    assert empty.eval_jvp(p, np.ones_like(p)).size == 0
    np.testing.assert_array_equal(empty.eval_vjp(p, np.zeros(0)), np.zeros_like(p))


def test_neural_mixed_derivatives(problem):
    (_, mapping, geometry, plastic, theta, _, _), topology = problem

    class NeuralDensity(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(22, 6), torch.nn.Tanh(), torch.nn.Linear(6, 1)
            ).double()

        def forward(self, F, P, t):
            return self.net(torch.cat([F.flatten(1), P.flatten(1), t], dim=1)).squeeze(
                -1
            )

    with torch.random.fork_rng():
        torch.manual_seed(42)
        material = NeuralDensity()
    u = np.linspace(-0.02, 0.03, 15)
    p = np.array([0.2, 0.4, 0.7, 0.8])
    fp = plastic[0].eval_value(np.full(3, 0.25))
    material_response = bridge.TorchVolumetricFemMaterialResponse(
        dof.ConstantDofMap(topology, 4),
        mapping,
        geometry,
        u,
        plastic,
        material=material,
    )
    fp_response = bridge.TorchVolumetricFemFpResponse(
        dof.ConstantDofMap(topology, 9),
        mapping,
        geometry,
        u,
        [AffineResponse(np.zeros((np.asarray(p).size, 3)), p)] * 2,
        material=material,
    )
    for model, x in ((material_response, p), (fp_response, fp)):
        J = model.eval_jacobian(x)
        v, w = np.linspace(0.1, 0.3, len(x)), np.linspace(-0.3, 0.4, 15)
        np.testing.assert_allclose(model.eval_jvp(x, v), J @ v, atol=1e-13)
        np.testing.assert_allclose(model.eval_vjp(x, w), J.T @ w, atol=1e-13)
        h = 1e-5
        np.testing.assert_allclose(
            model.eval_jvp(x, v),
            (model.eval_value(x + h * v) - model.eval_value(x - h * v)) / (2 * h),
            atol=1e-10,
            rtol=1e-7,
        )
    assert all(p.grad is None for p in material.parameters())


def test_equilibrium_response_example():
    import subprocess
    import sys

    path = (
        Path(__file__).resolve().parents[2]
        / "examples/fem/fitting/torch_equilibrium_response.py"
    )
    result = subprocess.run(
        [sys.executable, str(path)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(result.stdout.strip().splitlines()) == 2
    for line in result.stdout.strip().splitlines():
        fields = dict(
            part.strip().split("=", 1) for part in line.split(":", 1)[1].split(",")
        )
        assert all(float(value) < 1e-10 for value in fields.values())


@pytest.mark.parametrize("kind", ["material", "fp"])
def test_transformed_equilibrium_products(problem, kind):
    E, R, p, _ = factories(problem, kind, True, products_only=True)
    native_E, native_R, _, _ = factories(problem, kind, False)
    transform = (
        energy.ExpResponse(2) if kind == "material" else fem.LogDiagonalFpResponse(3)
    )
    q = np.log(p) if kind == "material" else np.log([1.04, 0.98, 1.02])
    physical = transform.eval_value(q)
    fixed = dof.DofReduction(15, [*range(9), 12, 13, 14])
    initial = np.zeros(15)
    initial[12:] = [0.04, -0.02, 0.06]
    options = solver.NewtonOptions(gradient_tolerance=1e-12, max_iterations=100)
    native = solver.StaticEquilibriumResponse(
        len(physical), fixed, initial, options, native_E, native_R
    )
    expected_J = native.eval_jacobian(physical) @ transform.eval_jacobian(q)
    obj = solver.StaticEquilibriumResponse(
        len(q),
        fixed,
        initial,
        options,
        lambda q: E(transform.eval_value(q)),
        lambda u: energy.ComposedResponse(R(u), transform),
    )
    v, w = np.linspace(0.1, 0.3, len(q)), np.linspace(-0.3, 0.4, 15)
    np.testing.assert_allclose(
        obj.eval_jvp(q, v), expected_J @ v, atol=1e-10, rtol=1e-9
    )
    np.testing.assert_allclose(
        obj.eval_vjp(q, w), expected_J.T @ w, atol=1e-10, rtol=1e-9
    )


@pytest.mark.parametrize("kind", ["material", "fp"])
def test_nonfinite_force_is_rejected_by_products(problem, kind):
    model, _, p, _, _ = responses(problem, kind, True, u=np.zeros(15))
    model.material = lambda F, P, t: torch.sqrt((F[:, 0, 0] - 1).square())
    for evaluate in (
        lambda: model.eval_value(p),
        lambda: model.eval_jvp(p, np.ones_like(p)),
        lambda: model.eval_vjp(p, np.ones(15)),
    ):
        with pytest.raises(ValueError):
            evaluate()


@pytest.mark.parametrize("p", [np.array([-1.0, 3.0]), np.array([2.0, 0.0])])
def test_native_material_domain_is_preserved(problem, p):
    model, native, _, _, _ = responses(problem, "material", True)
    for obj in (model, native):
        for evaluate in (
            lambda: obj.eval_value(p),
            lambda: obj.eval_jvp(p, np.ones(2)),
            lambda: obj.eval_vjp(p, np.ones(15)),
        ):
            with pytest.raises(ValueError):
                evaluate()


def test_large_unused_parameters_do_not_overflow_zero_derivatives(problem):
    model, _, p, _, _ = responses(problem, "material", True)
    model.material = lambda F, P, t: F.square().sum(dim=(1, 2))
    expected = model.eval_value(p)
    p = np.full_like(p, 1e200)
    np.testing.assert_allclose(model.eval_value(p), expected)
    np.testing.assert_array_equal(model.eval_jacobian(p).toarray(), 0)
    np.testing.assert_array_equal(model.eval_jvp(p, np.ones_like(p)), 0)
    np.testing.assert_array_equal(model.eval_vjp(p, np.ones(15)), 0)
