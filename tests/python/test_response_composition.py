from litepgo.energy import AffineResponse
import gc
import weakref

import numpy as np
import pytest
from litepgo import dof, energy, fem, mesh, solver


def topology():
    return mesh.TetTopology(5, np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32))


def check_products(response, x):
    J = response.evaluate(x).jacobian().toarray()
    v = np.linspace(0.2, 0.7, len(x))
    w = np.linspace(-0.3, 0.8, response.num_outputs)
    np.testing.assert_allclose(response.evaluate(x).jvp(v), J @ v, atol=1e-11)
    np.testing.assert_allclose(response.evaluate(x).vjp(w), J.T @ w, atol=1e-11)
    h = 1e-6
    numerical = (response.evaluate(x + h * v).value - response.evaluate(x - h * v).value) / (
        2 * h
    )
    np.testing.assert_allclose(J @ v, numerical, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("size", [2, 3])
def test_native_transforms(size):
    x = np.linspace(-0.2, 0.3, size)
    transform = fem.LogDiagonalFpResponse(size)
    expected = np.eye(3)
    expected[np.arange(size), np.arange(size)] = np.exp(x)
    np.testing.assert_allclose(transform.evaluate(x).value.reshape(3, 3, order='F'), expected)
    for response in [transform, energy.ExpResponse(size)]:
        check_products(response, x)
        for bad in [np.full(size, 1000.0), np.full(size, -1000.0)]:
            with pytest.raises(ValueError):
                response.evaluate(bad).value
        check_products(response, x)
    with pytest.raises(ValueError):
        fem.LogDiagonalFpResponse(1)
    with pytest.raises(ValueError):
        energy.ExpResponse(0)


@pytest.mark.parametrize("shared", [True, False])
def test_elementwise_values_and_shared_vjp(shared):
    mapping = (dof.ConstantDofMap if shared else dof.ElementDofMap)(topology(), 2)
    response = energy.ElementwiseResponse(mapping, energy.ExpResponse(2))
    x = np.linspace(-0.2, 0.3, mapping.num_dofs)
    expected = np.tile(np.exp(x), 2) if shared else np.exp(x)
    np.testing.assert_allclose(response.evaluate(x).value, expected)
    np.testing.assert_allclose(response.evaluate(x).vjp(np.ones(4)), (2 if shared else 1) * np.exp(x))
    check_products(response, x)
    with pytest.raises(ValueError):
        energy.ElementwiseResponse(mapping, energy.ExpResponse(3))


def test_direct_products_do_not_request_child_jacobians():
    class Direct(energy.FixedPatternResponse):
        def __init__(self):
            super().__init__(np.eye(2))

        def eval_impl(self, x, need_jacobian):
            if need_jacobian:
                raise AssertionError("explicit Jacobian requested")
            return x * x

        def jvp_at(self, x, v):
            return 2 * x * v

        def vjp_at(self, x, w):
            return 2 * x * w

    local = energy.ComposedResponse(Direct(), Direct())
    response = energy.ElementwiseResponse(dof.ConstantDofMap(topology(), 2), local)
    x = np.array([0.3, -0.5])
    np.testing.assert_allclose(response.evaluate(x).jvp(np.ones(2)), np.tile(4 * x ** 3, 2))
    np.testing.assert_allclose(response.evaluate(x).vjp(np.ones(4)), 8 * x ** 3)
    with pytest.raises(AssertionError, match="explicit"):
        response.evaluate(x).jacobian()
    np.testing.assert_allclose(response.evaluate(x).vjp(np.ones(4)), 8 * x ** 3)

    torch = pytest.importorskip("torch")
    from litepgo import torch as bridge

    parameters = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    bridge.response(response, parameters).sum().backward()
    np.testing.assert_allclose(parameters.grad.numpy(), 8 * x**3)
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(
            parameters.detach(), torch.ones_like(parameters)
        )
        tangent = torch.autograd.forward_ad.unpack_dual(
            bridge.response(response, dual)
        ).tangent
        np.testing.assert_allclose(tangent.numpy(), np.tile(4 * x**3, 2))


def test_dynamic_children_and_external_pattern_updates():
    class Dynamic(energy.DynamicPatternResponse):
        def __init__(self):
            super().__init__(2, 2)

        def jacobian_pattern_at(self, x):
            return np.array([[1.0, 0.0], [float(x[0] > 0), 1.0]])

        def eval_impl(self, x, need_jacobian):
            y = np.array([x[0], x[1] + max(x[0], 0.0) ** 2])
            J = np.array([[1.0, 0.0], [2 * max(x[0], 0.0), 1.0]])
            return (y, J) if need_jacobian else y

    child = Dynamic()
    composed = energy.ComposedResponse(child, energy.ExpResponse(2))
    x = np.array([0.3, -0.4])
    assert composed.evaluate(x).jacobian().shape == (2, 2)
    check_products(composed, x)
    check_products(composed, x)
    element = energy.ElementwiseResponse(dof.ElementDofMap(topology(), 2), child)
    check_products(element, np.array([0.2, 0.3, -0.4, 0.5]))
    with pytest.raises(ValueError):
        energy.ComposedResponse(energy.ExpResponse(3), energy.ExpResponse(2))


@pytest.mark.parametrize("kind", ["composed", "elementwise"])
def test_child_lifetime_with_weak_back_reference(kind):
    class Child(energy.FixedPatternResponse):
        def __init__(self):
            super().__init__(np.eye(2))

        def eval_impl(self, x, need_jacobian):
            return (x, np.eye(2)) if need_jacobian else x

    child = Child()
    if kind == "composed":
        owner = energy.ComposedResponse(child, energy.ExpResponse(2))
    else:
        owner = energy.ElementwiseResponse(dof.ConstantDofMap(topology(), 2), child)
    child.owner = weakref.ref(owner)
    assert child.owner() is owner
    owner_ref = weakref.ref(owner)
    ref = weakref.ref(child)
    del child
    gc.collect()
    owner.evaluate(np.ones(2)).value
    assert ref() is not None
    del owner
    gc.collect()
    assert ref() is None
    assert owner_ref() is None


def test_empty_mesh():
    empty = mesh.TetTopology(0, np.empty((0, 4), dtype=np.int32))
    obj = energy.ElementwiseResponse(
        dof.ConstantDofMap(empty, 2), energy.ExpResponse(2)
    )
    assert obj.evaluate(np.zeros(2)).value.size == 0
    assert obj.evaluate(np.zeros(2)).jacobian().shape == (0, 2)
    np.testing.assert_array_equal(obj.evaluate(np.zeros(2)).vjp(np.empty(0)), np.zeros(2))


@pytest.mark.parametrize("plastic", [True, False])
@pytest.mark.parametrize("shared", [True, False])
def test_fem_parameterizations_match_energy_and_equilibrium(plastic, shared):
    from test_material_force_response import problem, Energy

    topo, mapping, geometry, u, fixed_fp = problem()
    width = 3 if plastic else 2
    physical_width = 9 if plastic else 2
    parameter_map = (dof.ConstantDofMap if shared else dof.ElementDofMap)(topo, width)
    local = fem.LogDiagonalFpResponse() if plastic else energy.ExpResponse(2)
    transform = energy.ElementwiseResponse(parameter_map, local)
    physical_map = dof.ElementDofMap(topo, physical_width)
    theta = [AffineResponse(np.zeros((np.asarray(np.array([2.0, 3.0])).size, 3)), np.array([2.0, 3.0])) for _ in range(2)]
    force_type = (
        fem.VolumetricFemForceFpResponse
        if plastic
        else fem.VolumetricFemForceMaterialResponse
    )[fem.TetP1Basis, fem.TetCentroidQuadrature]

    def make_energy(p):
        fields = [
            AffineResponse(np.zeros((np.asarray(row).size, 3)), row)
            for row in transform.evaluate(p).value.reshape(2, physical_width)
        ]
        return Energy(
            fem.PointwiseVolumetricAdapter[fem.NeoHookean](), mapping,
            geometry,
            fields if plastic else fixed_fp,
            theta if plastic else fields,
        )

    def make_response(state):
        force = force_type(
            fem.PointwiseVolumetricAdapter[fem.NeoHookean](), physical_map, mapping, geometry, state, theta if plastic else fixed_fp
        )
        return energy.ComposedResponse(force, transform)

    p = np.linspace(-0.02, 0.03, parameter_map.num_dofs)
    force = make_response(u)
    np.testing.assert_allclose(force.evaluate(p).value, make_energy(p).eval_gradient(u), atol=1e-12)
    check_products(force, p)
    eq = solver.StaticEquilibriumResponse(
        parameter_map.num_dofs,
        dof.DofReduction(len(u), list(range(12))),
        u,
        solver.NewtonOptions(gradient_tolerance=1e-10, max_iterations=100),
        make_energy,
        make_response,
    )
    check_products(eq, p)


def test_composition_reuses_dynamic_child_at_different_inputs():
    class Flip(energy.DynamicPatternResponse):
        def __init__(self):
            super().__init__(2, 2)

        def jacobian_pattern_at(self, x):
            return np.array([[1.0, 0.0], [float(x[0] > 0), 1.0]])

        def eval_impl(self, x, need_jacobian):
            y = np.array([-x[0], x[1] + max(x[0], 0.0) ** 2])
            J = np.array([[-1.0, 0.0], [2 * max(x[0], 0.0), 1.0]])
            return (y, J) if need_jacobian else y

    child = Flip()
    obj = energy.ComposedResponse(child, child)
    for x in [np.array([0.3, 0.4]), np.array([-0.3, 0.4])]:
        # Flip composed with itself is (x0, x1+x0²), on either side of zero.
        np.testing.assert_allclose(obj.evaluate(x).jacobian().toarray(), [[1.0, 0.0], [2 * x[0], 1.0]])
        check_products(obj, x)
