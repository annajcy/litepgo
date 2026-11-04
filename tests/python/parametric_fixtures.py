"""Analytic formula adapter used only by tests."""

import numpy as np
from litepgo import energy


class FormulaModel(energy.ParametricEnergy):
    def __init__(self, n, size, factory, partial):
        super().__init__(np.ones((size, size)), np.ones((size, n)))
        self.factory, self.partial = factory, partial

    def eval_state_values(self, u, p, need_gradient, hessian_kind):
        potential = self.factory(p)
        if potential is None:
            raise ValueError("invalid energy")
        return potential.eval(u, gradient=need_gradient, hessian=hessian_kind)

    def eval_parameter_jacobian(self, u, p):
        return self.partial(p, u)

    def eval_feasible_step(self, u, p, du):
        return self.factory(p).max_feasible_step(u, du)
