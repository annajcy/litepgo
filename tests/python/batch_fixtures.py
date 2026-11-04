"""Test-only adapters for reference formulas and deliberate contract violations."""
from litepgo import energy


class BatchEnergyFixture(energy.MeshEnergy):
    def __init__(self, mapping, callback, batch_size=256, step=None):
        super().__init__(mapping, batch_size)
        self.callback, self.step = callback, step

    def eval_batch(self, ids, x, need_gradient, need_hessian):
        return self.callback(ids, x, need_gradient, need_hessian)

    def eval_feasible_step_batch(self, ids, x, dx):
        if self.step is None:
            return super().eval_feasible_step_batch(ids, x, dx)
        return self.step(ids, x, dx)


class BatchResponseFixture(energy.MeshResponse):
    def __init__(self, inputs, outputs, callback, batch_size=256):
        super().__init__(inputs, outputs, batch_size)
        self.callback = callback

    def eval_batch(self, ids, x, need_jacobian):
        return self.callback(ids, x, need_jacobian)
