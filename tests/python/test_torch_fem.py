from litepgo.energy import AffineResponse
from pathlib import Path
import runpy

import numpy as np
import pytest
import torch
from litepgo import dof, energy, fem, mesh, solver
from litepgo import torch as bridge

EXAMPLE = (
    Path(__file__).resolve().parents[2] / "examples/fem/fitting/torch_neohookean.py"
)


@pytest.fixture
def problem():
    return runpy.run_path(str(EXAMPLE))["make_problem"]()


@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_native_energy_force_and_tangent(problem, batch_size):
    nodes, mapping, geometry, plastic, theta, native, original = problem
    model = bridge.TorchVolumetricFemEnergy(
        mapping,
        geometry,
        plastic,
        theta,
        material=original.material,
        batch_size=batch_size,
    )
    rng = np.random.default_rng(16)
    for u in (
        np.zeros(nodes.size),
        rng.normal(scale=0.025, size=nodes.size),
        (-0.5 * nodes).ravel(),
    ):
        assert model.eval_value(u) == pytest.approx(native.eval_value(u), abs=1e-14)
        np.testing.assert_allclose(
            model.eval_gradient(u), native.eval_gradient(u), rtol=1e-12, atol=1e-13
        )
        for kind in (energy.HessianKind.Exact, energy.HessianKind.Projected):
            for want_gradient in (True, False):
                E, g, H = model.eval(u, gradient=want_gradient, hessian=kind)
                e, expected_g, expected_H = native.eval(
                    u, gradient=want_gradient, hessian=kind
                )
                assert E == pytest.approx(e, abs=1e-14)
                if want_gradient:
                    np.testing.assert_allclose(g, expected_g, rtol=1e-12, atol=1e-13)
                else:
                    assert g is None
                np.testing.assert_array_equal(H.indptr, expected_H.indptr)
                np.testing.assert_array_equal(H.indices, expected_H.indices)
                np.testing.assert_allclose(
                    H.toarray(), expected_H.toarray(), rtol=1e-11, atol=1e-12
                )


def test_finite_differences_and_torch_bridge(problem):
    *_, model = problem
    u = np.linspace(-0.015, 0.025, model.num_dofs)
    h = 1e-6
    direction = np.random.default_rng(7).normal(size=u.size)
    g, H = model.eval_gradient(u), model.eval_hessian(u)
    numerical_g = (
        model.eval_value(u + h * direction) - model.eval_value(u - h * direction)
    ) / (2 * h)
    numerical_H = (
        model.eval_gradient(u + h * direction) - model.eval_gradient(u - h * direction)
    ) / (2 * h)
    np.testing.assert_allclose(g @ direction, numerical_g, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(H @ direction, numerical_H, rtol=1e-7, atol=1e-9)
    with torch.no_grad():
        np.testing.assert_allclose(model.eval_gradient(u), g)
    with torch.inference_mode():
        np.testing.assert_allclose(model.eval_hessian(u).toarray(), H.toarray())
    x = torch.tensor(u, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: bridge.energy(model, x), (x,))


def test_feasible_step_and_equilibrium(problem):
    nodes, _, _, _, _, native, model = problem
    u = np.zeros(nodes.size)
    # All-axis collapse and a path whose endpoint has positive det but crosses zero.
    for A in (-2 * np.eye(3), np.diag([-3.0, -2.0, 0.0])):
        dx = (nodes @ A.T).ravel()
        step = model.max_feasible_step(u, dx)
        assert 0 < step < 1
        assert step == pytest.approx(native.max_feasible_step(u, dx), abs=1e-14)
        assert np.isfinite(model.eval_value(u + step * dx))
    fixed = dof.DofReduction(15, [*range(9), 12, 13, 14])
    u[12:] = [0.04, -0.02, 0.06]
    options = solver.NewtonOptions(gradient_tolerance=1e-10, max_iterations=100)
    a = solver.newton_solve(model, fixed, u, options)
    b = solver.newton_solve(native, fixed, u, options)
    assert a.status == b.status == solver.NewtonStatus.Converged
    np.testing.assert_array_equal(a.x[fixed.fixed_dofs], u[fixed.fixed_dofs])
    np.testing.assert_allclose(a.x, b.x, atol=1e-10, rtol=1e-9)
    assert np.max(np.abs(model.eval_gradient(a.x)[fixed.free_dofs])) < 1e-10


def test_field_sampling_and_input_validation(problem):
    nodes, mapping, geometry, plastic, theta, _, original = problem

    def create(**kw):
        return bridge.TorchVolumetricFemEnergy(
            mapping, geometry, plastic, theta, material=original.material, **kw
        )

    with pytest.raises((ValueError, RuntimeError)):
        create(batch_size=0)
    with pytest.raises(ValueError, match="one geometry"):
        bridge.TorchVolumetricFemEnergy(
            mapping, geometry[:1], plastic, theta, material=original.material
        )
    with pytest.raises(ValueError, match="component count"):
        bridge.TorchVolumetricFemEnergy(
            mapping,
            geometry,
            plastic,
            [theta[0], AffineResponse(np.zeros((np.asarray(np.ones(3)).size, 3)), np.ones(3))],
            material=original.material,
        )
    bad = AffineResponse(np.zeros((np.asarray(np.diag([-1.0, 1.0, 1.0]).ravel()).size, 3)), np.diag([-1.0, 1.0, 1.0]).ravel())
    with pytest.raises(ValueError, match="positive determinant"):
        bridge.TorchVolumetricFemEnergy(
            mapping, geometry, [bad] * 2, theta, material=original.material
        )
    with pytest.raises(ValueError, match="infeasible"):
        original.eval_value((-2 * nodes).ravel())
    # Nonconstant fields are sampled at the centroid, with the same native convention.
    Field = fem.InterpolatedResponse[fem.TetP1Basis]
    theta = [Field(np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]))] * 2
    model = bridge.TorchVolumetricFemEnergy(
        mapping, geometry, plastic, theta, material=original.material
    )
    native = fem.VolumetricFemEnergy[
        fem.NeoHookean, fem.TetP1Basis, fem.TetCentroidQuadrature
    ](mapping, geometry, plastic, theta)
    np.testing.assert_allclose(
        model.eval_hessian(np.zeros(15)).toarray(),
        native.eval_hessian(np.zeros(15)).toarray(),
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "material",
    [
        lambda F, Fp, t: torch.ones(len(F), dtype=torch.float64),
        lambda F, Fp, t: F.sum(dim=(1, 2)),
    ],
)
def test_constant_and_linear_materials(problem, material):
    _, mapping, geometry, plastic, theta, _, _ = problem
    model = bridge.TorchVolumetricFemEnergy(
        mapping, geometry, plastic, theta, material=material
    )
    assert np.linalg.norm(model.eval_hessian(np.zeros(15)).toarray()) == 0


@pytest.mark.parametrize(
    "material",
    [
        lambda F, Fp, t: F.sum(),
        lambda F, Fp, t: F.sum(dim=(1, 2)).float(),
        lambda F, Fp, t: np.zeros(len(F)),
        lambda F, Fp, t: F.sum(dim=(1, 2)) * float("nan"),
    ],
)
def test_bad_material_outputs(problem, material):
    _, mapping, geometry, plastic, theta, _, _ = problem
    model = bridge.TorchVolumetricFemEnergy(
        mapping, geometry, plastic, theta, material=material
    )
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        model.eval_value(np.zeros(15))


def test_neural_density_and_empty_mesh(problem):
    _, mapping, geometry, plastic, theta, _, _ = problem

    class NeuralDensity(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(9, 4), torch.nn.Softplus(), torch.nn.Linear(4, 1)
            ).double()

        def forward(self, F, Fp, theta):
            return self.net((F @ torch.linalg.inv(Fp)).reshape(-1, 9)).squeeze(-1)

    material = NeuralDensity()
    model = bridge.TorchVolumetricFemEnergy(
        mapping, geometry, plastic, theta, material=material
    )
    u = np.zeros(15)
    direction = np.linspace(-0.1, 0.2, 15)
    h = 1e-5
    np.testing.assert_allclose(
        model.eval_hessian(u) @ direction,
        (
            model.eval_gradient(u + h * direction)
            - model.eval_gradient(u - h * direction)
        )
        / (2 * h),
        atol=1e-9,
        rtol=1e-7,
    )
    assert all(p.grad is None for p in material.parameters())
    empty = dof.VertexDofMap(mesh.TetTopology(0, np.empty((0, 4), dtype=np.int32)), 3)
    model = bridge.TorchVolumetricFemEnergy(empty, [], [], [], material=material)
    assert model.eval_value(np.zeros(0)) == 0


def test_geometry_map_and_snapshot_contract(problem):
    _, mapping, geometry, plastic, theta, _, original = problem
    u = np.zeros(15)
    value = original.eval_value(u)
    # No field-list lifetime dependency after construction.
    plastic.clear()
    theta.clear()
    geometry.clear()
    assert original.eval_value(u) == value
    with pytest.raises(TypeError, match="VertexDofMap"):
        bridge.TorchVolumetricFemEnergy(
            dof.StencilDofMap(4, [[0, 1, 2, 3]], 3),
            [],
            [],
            [],
            material=original.material,
        )
    hexmap = dof.VertexDofMap(
        mesh.HexTopology(8, np.arange(8, dtype=np.int32).reshape(1, 8)), 3
    )
    with pytest.raises(ValueError, match="TetP1"):
        bridge.TorchVolumetricFemEnergy(hexmap, [], [], [], material=original.material)
    Field = fem.InterpolatedResponse[fem.TetP1Basis]
    for nodes in (
        np.zeros((4, 3)),
        np.array([[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    ):
        with pytest.raises(ValueError, match="positive geometry determinant"):
            bridge.TorchVolumetricFemEnergy(
                mapping,
                [Field(nodes)] * 2,
                [AffineResponse(np.zeros((np.asarray(np.eye(3).ravel()).size, 3)), np.eye(3).ravel())] * 2,
                [AffineResponse(np.zeros((np.asarray(np.ones(2)).size, 3)), np.ones(2))] * 2,
                material=original.material,
            )


def test_nonfinite_derivatives_reject_trial(problem):
    _, mapping, geometry, plastic, theta, _, _ = problem

    def nonsmooth(F, Fp, theta):
        return torch.sqrt((F[:, 0, 0] - 1).square())

    model = bridge.TorchVolumetricFemEnergy(
        mapping, geometry, plastic, theta, material=nonsmooth
    )
    assert model.eval_value(np.zeros(15)) == 0
    with pytest.raises(ValueError, match="infeasible"):
        model.eval_gradient(np.zeros(15))


def test_example_runs(capsys):
    runpy.run_path(str(EXAMPLE), run_name="__main__")
    output = capsys.readouterr().out
    assert (
        "Exact:" in output and "Projected:" in output and "equilibrium_error=" in output
    )


def test_large_shear_constant_density_has_zero_derivatives(problem):
    nodes, mapping, geometry, plastic, theta, _, _ = problem
    model = bridge.TorchVolumetricFemEnergy(
        mapping,
        geometry,
        plastic,
        theta,
        material=lambda F, P, t: torch.ones(len(F), dtype=torch.float64),
    )
    u = np.zeros_like(nodes)
    u[:, 0] = 1e200 * nodes[:, 1]
    assert np.isfinite(model.eval_value(u.ravel()))
    np.testing.assert_array_equal(model.eval_gradient(u.ravel()), 0)
    np.testing.assert_array_equal(model.eval_hessian(u.ravel()).toarray(), 0)


def test_skew_geometry_sampling_matches_native_fem(problem):
    _, mapping, _, plastic, theta, _, original = problem
    J = np.array([[2., .3, -.2], [.1, 1.5, .4], [.2, -.1, 3.]])
    geometry = [energy.AffineResponse(J)] * mapping.num_elements
    model = bridge.TorchVolumetricFemEnergy(mapping, geometry, plastic, theta, material=original.material)
    native = fem.VolumetricFemEnergy[fem.NeoHookean, fem.TetP1Basis, fem.TetCentroidQuadrature](
        mapping, geometry, plastic, theta
    )
    u = np.random.default_rng(4).normal(scale=.02, size=mapping.num_dofs)
    assert model.eval_value(u) == pytest.approx(native.eval_value(u), abs=1e-13)
    np.testing.assert_allclose(model.eval_gradient(u), native.eval_gradient(u), atol=1e-13)
    np.testing.assert_allclose(model.eval_hessian(u).toarray(), native.eval_hessian(u).toarray(), atol=1e-12)
