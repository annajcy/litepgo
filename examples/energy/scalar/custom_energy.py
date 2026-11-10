"""Subclass fixed/dynamic Energy; request values, gradients and sparse Hessians.

Run from the repository root after building litepgo:
    uv run --no-sync python examples/energy/scalar/custom_energy.py
"""

import numpy as np
from litepgo import energy


class Quadratic(energy.FixedPatternEnergy):
    def __init__(self):
        super().__init__(np.eye(2))

    def eval_values(self, x, want_gradient, need_hessian):
        r = x - [1.0, 2.0]
        return (
            0.5 * r @ r,
            r if want_gradient else None,
            np.eye(2) if need_hessian else None,
        )


class Coupled(energy.DynamicPatternEnergy):
    def __init__(self):
        super().__init__(2)

    def hessian_pattern_at(self, x):
        return np.ones((2, 2)) if x.sum() > 0 else np.eye(2)

    def eval_values(self, x, want_gradient, need_hessian):
        s = max(0.0, x.sum())
        # This convex energy has identical Exact and Projected Hessians.
        return (
            0.5 * x @ x + 0.25 * s**4,
            x + s**3 if want_gradient else None,
            np.eye(2) + 3 * s * s if need_hessian else None,
        )


def main():
    fixed, dynamic = Quadratic(), Coupled()
    x = np.ones(2)
    fixed_evaluation = fixed.evaluate(x)
    H = fixed_evaluation.hessian()
    g = fixed_evaluation.gradient()
    value = fixed_evaluation.value
    active_value = dynamic.evaluate(x).value  # Request only the value.
    dynamic_evaluation = dynamic.evaluate(x)
    H = dynamic_evaluation.hessian()
    g = dynamic_evaluation.gradient()
    active_nnz = H.nnz
    H = dynamic.evaluate(-x).hessian()
    print(
        f"fixed value={value:g}, active value={active_value:g}, Hessian nnz=2->{active_nnz}->{H.nnz}"
    )


if __name__ == "__main__":
    main()
