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
