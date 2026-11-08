"""Fixed-pose force formulas for equilibrium tests."""

import numpy as np
from litepgo import energy


class FormulaForce(energy.FixedPatternResponse):
    def __init__(self, n, u, make_energy, partial):
        super().__init__(np.ones((len(u), n)))
        self.u = u.copy()
        self.make_energy, self.partial = make_energy, partial

    def eval_impl(self, p, need_jacobian):
        g = self.make_energy(p).eval_gradient(self.u)
        return (g, self.partial(p, self.u)) if need_jacobian else g
