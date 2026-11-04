import gc
import weakref

import numpy as np
import pytest
from litepgo import dof, energy, solver


class Quadratic(energy.ParametricEnergy):
    def __init__(self):
        self.K = np.array([[4.0, 1.0, 0.5], [1.0, 3.0, -0.4], [0.5, -0.4, 2.0]])
        self.A = np.array([[1.0, -0.5], [0.3, 2.0], [-0.7, 1.2]])
        super().__init__(np.ones((3, 3)), np.ones((3, 2)))
        self.kinds = []

    def eval_state_values(self, u, p, need_gradient, hessian_kind):
        self.kinds.append(hessian_kind)
        return (
            0.5 * u @ self.K @ u - u @ self.A @ p,
            self.K @ u - self.A @ p if need_gradient else None,
            self.K if hessian_kind is not None else None,
        )

    def eval_parameter_jacobian(self, u, p):
        return -self.A

    def eval_feasible_step(self, u, p, du):
        return 0.75


def test_views_copy_inputs_and_share_gradient_and_projection():
    model = Quadratic()
    p, u = np.array([0.2, -0.7]), np.array([0.1, 0.3, -0.4])
    p0, u0 = p.copy(), u.copy()
    potential, force = model.energy_at(p), model.force_at(u)
    p[:] = 99
    u[:] = 99
    np.testing.assert_allclose(potential.eval_gradient(u0), force.eval_value(p0))
    np.testing.assert_allclose(force.eval_jacobian(p0).toarray(), -model.A)
    np.testing.assert_allclose(force.eval_jvp(p0, p0), -model.A @ p0)
    np.testing.assert_allclose(force.eval_vjp(p0, u0), -model.A.T @ u0)
    potential.eval_projected_hessian(u0)
    assert model.kinds[-1] == energy.HessianKind.Projected
    assert potential.max_feasible_step(u0, u0) == 0.75
    assert model.num_dofs == 3 and model.num_parameters == 2
    for call, arg in [
        (model.energy_at, np.zeros(3)),
        (model.force_at, np.full(3, np.nan)),
    ]:
        with pytest.raises(ValueError):
            call(arg)


def test_equilibrium_fast_paths_do_not_request_B():
    class Direct(Quadratic):
        def eval_parameter_jacobian(self, u, p):
            raise AssertionError("explicit B requested")

        def eval_parameter_jvp(self, u, p, v):
            return -self.A @ v

        def eval_parameter_vjp(self, u, p, w):
            return -self.A.T @ w

        def eval_feasible_step(self, u, p, du):
            return 1.0

    model = Direct()
    eq = solver.StaticEquilibriumResponse(
        model,
        dof.DofReduction(3, [1]),
        np.array([0.0, 0.8, 0.0]),
        solver.NewtonOptions(gradient_tolerance=1e-12),
    )
    p, v, w = np.array([0.6, -1.1]), np.array([0.3, -0.7]), np.array([0.4, 7.0, -0.8])
    J = np.zeros((3, 2))
    J[[0, 2]] = np.linalg.solve(model.K[np.ix_([0, 2], [0, 2])], model.A[[0, 2]])
    np.testing.assert_allclose(eq.eval_jvp(p, v), J @ v, atol=1e-12)
    np.testing.assert_allclose(eq.eval_vjp(p, w), J.T @ w, atol=1e-12)
    h = 1e-5
    numerical = (eq.eval_value(p + h * v) - eq.eval_value(p - h * v)) / (2 * h)
    np.testing.assert_allclose(eq.eval_jvp(p, v), numerical, atol=1e-9)
    with pytest.raises(AssertionError, match="explicit B"):
        eq.eval_jacobian(p)
    np.testing.assert_allclose(eq.eval_vjp(p, w), J.T @ w, atol=1e-12)


def test_dynamic_patterns_are_owned_by_each_view():
    class Dynamic(energy.ParametricEnergy):
        def __init__(self):
            super().__init__(np.eye(2), np.eye(2), fixed_pattern=False)

        def state_pattern_at(self, u, p):
            return np.array([[1.0, float(p[0] > 0)], [float(p[0] > 0), 1.0]])

        def parameter_pattern_at(self, u, p):
            return np.array([[1.0, 0.0], [float(p[0] > 0), 1.0]])

        def eval_state_values(self, u, p, g, kind):
            # E = .5 |u-p|² + max(p0,0) u0 u1.
            a = max(p[0], 0.0)
            K = np.array([[1.0, a], [a, 1.0]])
            return (
                0.5 * np.dot(u - p, u - p) + a * u[0] * u[1],
                K @ u - p if g else None,
                K if kind is not None else None,
            )

        def eval_parameter_jacobian(self, u, p):
            a = float(p[0] > 0)
            return np.array([[-1.0 + a * u[1], 0.0], [a * u[0], -1.0]])

    model = Dynamic()
    pos, neg, u = np.array([0.3, 0.4]), np.array([-0.3, 0.4]), np.array([0.2, 0.7])
    ep, en = model.energy_at(pos), model.energy_at(neg)
    fp, fn = model.force_at(u), model.force_at(u)
    ep.update_pattern(u)
    en.update_pattern(u)
    fp.update_pattern(pos)
    fn.update_pattern(neg)
    assert ep.hessian_pattern.nnz == 4 and en.hessian_pattern.nnz == 2
    assert fp.jacobian_pattern.nnz == 3 and fn.jacobian_pattern.nnz == 2
    for obj, p in [(fp, pos), (fn, neg)]:
        J = obj.eval_jacobian(p)
        np.testing.assert_allclose(obj.eval_jvp(p, p), J @ p)
        np.testing.assert_allclose(obj.eval_vjp(p, u), J.T @ u)
    assert (
        fp.jacobian_pattern.nnz == 3
    )  # Evaluating another view never mutates this one.


@pytest.mark.parametrize("view", ["energy", "force", "equilibrium"])
def test_native_owners_keep_python_model_alive_and_collect_cycles(view):
    model = Quadratic()
    if view == "energy":
        obj = model.energy_at(np.ones(2))
    elif view == "force":
        obj = model.force_at(np.zeros(3))
    else:
        obj = solver.StaticEquilibriumResponse(
            model,
            dof.DofReduction(3, []),
            np.zeros(3),
            solver.NewtonOptions(gradient_tolerance=1e-10),
        )
    model.view = obj
    ref = weakref.ref(model)
    del model
    gc.collect()
    assert ref() is not None
    obj.eval_value(np.ones(3 if view == "energy" else 2))
    del obj
    gc.collect()
    assert ref() is None


def test_model_guard_covers_different_views_and_recovers():
    class Reentrant(Quadratic):
        nested = False

        def eval_state_values(self, u, p, g, kind):
            if self.nested:
                self.force_at(u).eval_value(p)
            return super().eval_state_values(u, p, g, kind)

    model = Reentrant()
    obj = model.energy_at(np.zeros(2))
    model.nested = True
    with pytest.raises(RuntimeError, match="ParametricEnergy"):
        obj.eval_value(np.zeros(3))
    model.nested = False
    assert obj.eval_value(np.zeros(3)) == 0.0


def test_product_super_fallback_and_invalid_outputs_recover():
    class Checked(Quadratic):
        mode = "ok"

        def eval_parameter_jacobian(self, u, p):
            if self.mode == "bad_B":
                return np.ones((1, 2))
            return super().eval_parameter_jacobian(u, p)

        def eval_parameter_jvp(self, u, p, v):
            if self.mode == "bad_product":
                return np.zeros(2)
            return super().eval_parameter_jvp(u, p, v)

        def eval_parameter_vjp(self, u, p, w):
            if self.mode == "nan":
                return np.full(2, np.nan)
            return super().eval_parameter_vjp(u, p, w)

    model = Checked()
    force = model.force_at(np.zeros(3))
    p = np.ones(2)
    for mode in ["bad_B", "bad_product"]:
        model.mode = mode
        with pytest.raises(RuntimeError):
            force.eval_jvp(p, p)
    model.mode = "nan"
    with pytest.raises(RuntimeError):
        force.eval_vjp(p, np.ones(3))
    model.mode = "ok"
    np.testing.assert_allclose(force.eval_jvp(p, p), -model.A @ p)
    np.testing.assert_allclose(force.eval_vjp(p, np.ones(3)), -model.A.T @ np.ones(3))
    with pytest.raises(ValueError):
        model.energy_at(np.full(2, np.inf))


def test_invalid_patterns_and_state_derivatives():
    with pytest.raises(ValueError):
        energy.ParametricEnergy(np.ones((2, 3)), np.ones((2, 1)))
    with pytest.raises(ValueError):
        energy.ParametricEnergy(np.eye(2), np.ones((3, 1)))

    class Bad(Quadratic):
        def eval_state_values(self, u, p, g, kind):
            return (
                0.0,
                np.ones(1) if g else None,
                np.ones((1, 1)) if kind is not None else None,
            )

    model = Bad()
    with pytest.raises(RuntimeError):
        model.energy_at(np.zeros(2)).eval_gradient(np.zeros(3))
    with pytest.raises(RuntimeError):
        model.force_at(np.zeros(3)).eval_value(np.zeros(2))
    with pytest.raises(RuntimeError):
        model.energy_at(np.zeros(2)).eval_hessian(np.zeros(3))
