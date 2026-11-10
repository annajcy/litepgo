from litepgo.solver import AbsoluteConvergence
import gc
import weakref
import numpy as np
import pytest
import scipy.sparse as sp
import torch
from litepgo import energy, solver, dof
from litepgo import torch as lt


def problem(kind=False):
    calls = dict(energy=0, force=0)
    K = sp.csc_matrix(np.diag([2., 3., 4.]))
    A = np.array([[1., .2], [.3, 1.], [.1, -.4]])
    def make_energy(p):
        calls['energy'] += 1
        return energy.QuadraticEnergy(K, -A @ p)
    def make_force(u):
        calls['force'] += 1
        return energy.AffineResponse(-A, K @ u)
    eq = solver.StaticEquilibriumResponse(2, dof.DofReduction(3, []), np.zeros(3),
        solver.NewtonOptions(project_hessian=kind), make_energy, make_force, convergence=AbsoluteConvergence(1e-12))
    return eq, calls, np.linalg.solve(K.toarray(), A)


@pytest.mark.parametrize('kind', [False, True])
def test_equilibrium_state_reuses_solve_and_force(kind):
    eq, calls, J = problem(kind)
    p = np.array([.3, .4])
    state = eq.evaluate(p)
    p[:] = 100
    np.testing.assert_allclose(state.value, J @ [.3, .4])
    assert calls == dict(energy=1, force=0)
    for _ in range(2):
        np.testing.assert_allclose(state.jvp(np.ones(2)), J @ np.ones(2))
        np.testing.assert_allclose(state.vjp(np.ones(3)), J.T @ np.ones(3))
        np.testing.assert_allclose(state.jacobian().toarray(), J)
    assert calls == dict(energy=1, force=1)
    with pytest.raises(RuntimeError): state.hessian(np.ones(3))


@pytest.mark.parametrize('composed', [False, True])
def test_torch_two_forwards_reverse_backward(composed):
    eq, calls, J = problem()
    model = energy.ComposedResponse(energy.ExpResponse(3), eq) if composed else eq
    p = torch.tensor([.3, .4], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([-.2, .1], dtype=torch.float64, requires_grad=True)
    y, z = lt.response(model, p), lt.response(model, q)
    z.sum().backward(); y.sum().backward()
    for x in (p, q):
        weights = np.exp(J @ x.detach().numpy()) if composed else np.ones(3)
        np.testing.assert_allclose(x.grad.numpy(), J.T @ weights)
    assert calls == dict(energy=2, force=2)


def test_lifetime_value_copy_and_removed_methods():
    obj = energy.AffineResponse(sp.eye(3, format='csc'))
    ref = weakref.ref(obj)
    state = obj.evaluate(np.arange(3.))
    for method in ('eval_value', 'eval_jacobian', 'eval_value_and_jacobian', 'eval_jvp', 'eval_vjp', 'eval_hessian'):
        assert not hasattr(obj, method)
    del obj; gc.collect()
    assert ref() is not None
    value = state.value
    value[:] = -1
    np.testing.assert_array_equal(state.value, np.arange(3.))
    np.testing.assert_array_equal(state.jvp(np.ones(3)), np.ones(3))
    del state; gc.collect()
    assert ref() is None


def test_evaluation_dimensions_and_finite_values():
    obj = energy.ExpResponse(3)
    for x in (np.zeros(2), np.full(3, np.nan)):
        with pytest.raises(ValueError): obj.evaluate(x)
    state = obj.evaluate(np.zeros(3))
    for product in (state.jvp, state.vjp, state.hessian):
        for w in (np.zeros(2), np.full(3, np.nan)):
            with pytest.raises(ValueError): product(w)

@pytest.mark.parametrize('kind', [False, True])
def test_final_hessian_is_deferred_until_sensitivity(kind):
    requests = []
    class Potential(energy.FixedPatternEnergy):
        def __init__(self, p):
            super().__init__(sp.eye(2, format='csc'))
            self.p = p.copy()
        def eval_values(self, x, gradient, need_hessian):
            if need_hessian: requests.append("exact")
            return (.5 * x @ x - self.p @ x,
                    x - self.p if gradient else None,
                    sp.eye(2, format='csc') if need_hessian else None)
        def eval_projected_values(self, x, gradient, need_hessian):
            if need_hessian: requests.append("projected")
            return (.5 * x @ x - self.p @ x,
                    x - self.p if gradient else None,
                    sp.eye(2, format='csc') if need_hessian else None)
    eq = solver.StaticEquilibriumResponse(2, dof.DofReduction(2, []), np.zeros(2),
        solver.NewtonOptions(project_hessian=kind), Potential,
        lambda u: energy.AffineResponse(-sp.eye(2, format='csc'), u), convergence=AbsoluteConvergence(1e-12))
    state = eq.evaluate(np.array([.3, .4]))
    np.testing.assert_allclose(state.value, [.3, .4])
    assert requests == ["projected" if kind else "exact"]
    state.vjp(np.ones(2))
    expected = ["projected" if kind else "exact"] + ["exact"]
    assert requests == expected
    state.jvp(np.ones(2)); state.jacobian()
    assert requests == expected


def test_composed_joint_request_reuses_child_values():
    calls = []
    class Map(energy.FixedPatternResponse):
        def __init__(self, name):
            super().__init__(sp.eye(2, format='csc'))
            self.name = name
        def eval_impl(self, x, need_jacobian):
            calls.append((self.name, need_jacobian))
            return (2*x, 2*sp.eye(2, format='csc')) if need_jacobian else 2*x
    model = energy.ComposedResponse(Map('outer'), Map('inner'))
    state = model.evaluate(np.ones(2))
    np.testing.assert_allclose(state.jacobian().toarray(), 4*np.eye(2))
    np.testing.assert_allclose(state.value, 4*np.ones(2))
    assert calls == [('inner', True), ('outer', True)]


def test_elementwise_retains_each_equilibrium_evaluation():
    from litepgo import mesh
    eq, calls, J = problem()
    topology = mesh.TetTopology(8, np.arange(8, dtype=np.int32).reshape(2, 4))
    model = energy.ElementwiseResponse(dof.ConstantDofMap(topology, 2), eq)
    p = np.array([.3, .4])
    state = model.evaluate(p)
    np.testing.assert_allclose(state.value, np.tile(J @ p, 2))
    assert calls == dict(energy=2, force=0)
    np.testing.assert_allclose(state.vjp(np.ones(6)), 2 * J.T @ np.ones(3))
    np.testing.assert_allclose(state.jvp(np.ones(2)), np.tile(J @ np.ones(2), 2))
    np.testing.assert_allclose(state.jacobian().toarray(), np.tile(J, (2, 1)))
    assert calls == dict(energy=2, force=2)
    del model, eq
    gc.collect()
    np.testing.assert_allclose(state.value, np.tile(J @ p, 2))
    np.testing.assert_allclose(state.vjp(np.ones(6)), 2 * J.T @ np.ones(3))
    assert calls == dict(energy=2, force=2)


@pytest.mark.parametrize('project', [False, True])
def test_equilibrium_backward_handles_different_projected_pattern(project):
    K = np.array([[2., .5], [.5, 3.]])

    class Potential(energy.FixedPatternEnergy):
        def __init__(self, p):
            super().__init__(sp.csc_matrix(K))
            self.p = p.copy()

        def eval_values(self, x, gradient, hessian):
            return (.5 * x @ K @ x - self.p @ x,
                    K @ x - self.p if gradient else None,
                    K if hessian else None)

        def eval_projected_values(self, x, gradient, hessian):
            value, g, _ = self.eval_values(x, gradient, False)
            return value, g, sp.eye(2, format='csc') if hessian else None

    eq = solver.StaticEquilibriumResponse(
        2, dof.DofReduction(2, []), np.zeros(2),
        solver.NewtonOptions(project_hessian=project),
        Potential, lambda u: energy.AffineResponse(-sp.eye(2, format='csc'), K @ u), convergence=AbsoluteConvergence(1e-12))
    state = eq.evaluate(np.zeros(2))
    np.testing.assert_allclose(state.value, np.zeros(2))
    direction = np.array([1., 2.])
    expected = np.linalg.solve(K, direction)
    np.testing.assert_allclose(state.jvp(direction), expected)
    np.testing.assert_allclose(state.vjp(direction), expected)
    np.testing.assert_allclose(state.jacobian().toarray(), np.linalg.inv(K))
