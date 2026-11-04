import numpy as np
import pytest
from scipy import sparse

import torch
from litepgo import dof, energy, solver
from litepgo import torch as bridge


class Square(energy.FixedPatternResponse):
    def __init__(self):
        super().__init__(sparse.eye(2, format="csc"))
        self.derivatives = 0

    def eval_values(self, x, need_jacobian):
        if need_jacobian:
            self.derivatives += 1
            return x * x, sparse.diags(2 * x, format="csc")
        return x * x


def test_energy_and_response_gradcheck_and_snapshots():
    x = torch.tensor([0.3, 0.7], dtype=torch.float64, requires_grad=True)
    potential = energy.QuadraticEnergy(sparse.eye(2, format="csc"), np.ones(2))
    assert torch.autograd.gradcheck(lambda x: bridge.energy(potential, x), (x,))
    model = Square()
    assert torch.autograd.gradcheck(lambda x: bridge.response(model, x), (x,))
    a = bridge.response(model, x)
    b = bridge.response(model, 2 * x)
    (a.sum() + b.sum()).backward()
    torch.testing.assert_close(x.grad, 10 * x)
    before = model.derivatives
    with torch.no_grad():
        bridge.response(model, x)
    assert before == model.derivatives
    with pytest.raises(ValueError):
        bridge.response(model, x.float())
    with pytest.raises(ValueError):
        bridge.response(model, x.reshape(1, 2))


def test_equilibrium_gradcheck_fixed_dofs_and_failure():
    model = solver.StaticEquilibriumResponse(
        2,
        dof.DofReduction(2, [0]),
        np.array([0.4, 0.0]),
        solver.NewtonOptions(gradient_tolerance=1e-12),
        lambda p: energy.QuadraticEnergy(sparse.eye(2, format="csc"), -p),
        lambda p, u: -sparse.eye(2, format="csc"),
    )
    p = torch.tensor([2.0, 3.0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda p: bridge.response(model, p), (p,))
    bridge.response(model, p).sum().backward()
    torch.testing.assert_close(p.grad, torch.tensor([0.0, 1.0], dtype=torch.float64))
    with pytest.raises(ValueError):
        bridge.response(model, torch.tensor([float("nan"), 1.0], dtype=torch.float64))


def test_double_backward_is_rejected():
    x = torch.ones(2, dtype=torch.float64, requires_grad=True)
    for f in (
        lambda x: bridge.response(Square(), x).sum(),
        lambda x: bridge.energy(energy.QuadraticEnergy(sparse.eye(2, format="csc")), x),
    ):
        first = torch.autograd.grad(f(x), x, create_graph=True)[0]
        with pytest.raises(RuntimeError):
            torch.autograd.grad(first.sum(), x)


def forward_product(model, x, tangent):
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(x, tangent)
        result = bridge.response(model, dual)
        return torch.autograd.forward_ad.unpack_dual(result)


def test_direct_products_skip_explicit_jacobian():
    class Direct(Square):
        def __init__(self):
            super().__init__()
            self.products = []

        def eval_values(self, x, need_jacobian):
            assert not need_jacobian, "explicit Jacobian requested"
            return x*x

        def eval_jvp_values(self, x, dx):
            self.products.append("jvp")
            return 2*x*dx

        def eval_vjp_values(self, x, gy):
            self.products.append("vjp")
            return 2*x*gy

    model = Direct()
    x = torch.tensor([.3, .7], dtype=torch.float64, requires_grad=True)
    value = bridge.response(model, x)
    assert model.products == []
    value.sum().backward(retain_graph=True)
    torch.testing.assert_close(x.grad, 2*x)
    assert model.products == ["vjp"]
    torch.testing.assert_close(torch.autograd.grad(value.sum(), x)[0], 2*x)
    assert model.products == ["vjp", "vjp"]
    model.products.clear()
    with torch.no_grad():
        bridge.response(model, x)
    assert model.products == []
    tangent = torch.tensor([.2, -.4], dtype=torch.float64)
    # Forward AD must work even with reverse AD disabled.
    with torch.no_grad():
        primal, product = forward_product(model, x.detach(), tangent)
    torch.testing.assert_close(primal, x*x)
    torch.testing.assert_close(product, 2*x*tangent)
    assert model.products == ["jvp"]
    assert torch.autograd.gradcheck(
        lambda x: bridge.response(model, x), (x,), check_forward_ad=True
    )


def test_dynamic_products_restore_saved_point_and_input_copy():
    from test_dynamic_response import Piecewise

    model = Piecewise()
    a = torch.tensor([.3, .4], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([-.3, .4], dtype=torch.float64, requires_grad=True)
    saved_a = a.detach().clone()
    ya, yb = bridge.response(model, a), bridge.response(model, b)
    model.update_pattern(b.detach().numpy())
    with torch.no_grad():
        a.add_(10)  # Forward owns an independent input copy.
    ga, gb = torch.autograd.grad(ya.sum() + yb.sum(), (a, b))
    for point, actual in [(saved_a, ga), (b.detach(), gb)]:
        model.update_pattern(point.numpy())
        expected = model.eval_jacobian(point.numpy()).T @ np.ones(2)
        np.testing.assert_allclose(actual.numpy(), expected)
        tangent = torch.tensor([.5, -.2], dtype=torch.float64)
        model.update_pattern(-point.numpy())  # Deliberately leave the opposite active pattern.
        _, product = forward_product(model, point, tangent)
        np.testing.assert_allclose(product.numpy(), model.eval_jvp(point.numpy(), tangent.numpy()))


def test_equilibrium_forward_ad_and_product_errors():
    calls = []

    def partial(p, u):
        calls.append(1)
        return -sparse.eye(2, format="csc")

    model = solver.StaticEquilibriumResponse(
        2, dof.DofReduction(2, [0]), np.array([.4, 0.]),
        solver.NewtonOptions(gradient_tolerance=1e-12),
        lambda p: energy.QuadraticEnergy(sparse.eye(2, format="csc"), -p), partial,
    )
    x = torch.tensor([2., 3.], dtype=torch.float64, requires_grad=True)
    y = bridge.response(model, x)
    assert not calls
    y.sum().backward()
    assert len(calls) == 1
    _, product = forward_product(model, x.detach(), torch.ones_like(x))
    torch.testing.assert_close(product, torch.tensor([0., 1.], dtype=torch.float64))
    assert len(calls) == 2

    class Failure(Square):
        def eval_vjp_values(self, x, gy):
            raise ValueError("product failure")
        def eval_jvp_values(self, x, dx):
            raise ValueError("product failure")

    with pytest.raises(ValueError, match="product failure"):
        bridge.response(Failure(), x).sum().backward()
    with pytest.raises(ValueError, match="product failure"):
        forward_product(Failure(), x, torch.ones_like(x))


def test_graph_retains_model_until_graph_is_released():
    import gc
    import weakref

    model = Square()
    reference = weakref.ref(model)
    x = torch.ones(2, dtype=torch.float64, requires_grad=True)
    y = bridge.response(model, x)
    del model
    gc.collect()
    assert reference() is not None
    y.sum().backward()
    torch.testing.assert_close(x.grad, 2*x)
    del y
    gc.collect()
    assert reference() is None


def test_rectangular_mesh_products_and_noncontiguous_tensors():
    A = np.array([[1., 2.], [-3., 4.], [5., 0.]])

    class Rectangular(energy.MeshResponse):
        def __init__(self):
            super().__init__(
                dof.StencilDofMap(2, [[0, 1]], 1),
                dof.StencilDofMap(3, [[0, 1, 2]], 1),
            )

        def eval_batch(self, ids, x, need):
            assert not need, "explicit Jacobian requested"
            return x @ A.T

        def eval_jvp_batch(self, ids, x, dx):
            return dx @ A.T

        def eval_vjp_batch(self, ids, x, gy):
            return gy @ A

    model = Rectangular()
    x = torch.arange(4, dtype=torch.float64)[::2].requires_grad_()
    w = torch.arange(6, dtype=torch.float64)[::2]
    tangent = torch.arange(4, dtype=torch.float64)[::2]
    assert not x.is_contiguous() and not w.is_contiguous()
    value = bridge.response(model, x)
    grad = torch.autograd.grad(value, x, grad_outputs=w)[0]
    np.testing.assert_allclose(grad.numpy(), A.T @ w.numpy())
    primal, jvp = forward_product(model, x.detach(), tangent)
    np.testing.assert_allclose(primal.numpy(), A @ x.detach().numpy())
    np.testing.assert_allclose(jvp.numpy(), A @ tangent.numpy())
