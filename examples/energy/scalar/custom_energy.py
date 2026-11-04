"""Subclass fixed/dynamic Energy; request values, gradients and sparse Hessians.

Run from the repository root after building litepgo:
    uv run --no-sync python examples/energy/scalar/custom_energy.py
"""

import numpy as np
from litepgo import energy


class Quadratic(energy.FixedPatternEnergy):
    def __init__(self):
        super().__init__(2, np.eye(2))

    def eval_values(self, x, want_gradient, hessian_kind):
        r = x - [1.0, 2.0]
        return (
            0.5 * r @ r,
            r if want_gradient else None,
            np.eye(2) if hessian_kind is not None else None,
        )


class Coupled(energy.DynamicPatternEnergy):
    def __init__(self):
        super().__init__(2, np.eye(2))

    def pattern_at(self, x):
        return np.ones((2, 2)) if x.sum() > 0 else np.eye(2)

    def eval_values(self, x, want_gradient, hessian_kind):
        s = max(0.0, x.sum())
        # This convex energy has identical Exact and Projected Hessians.
        return (
            0.5 * x @ x + 0.25 * s**4,
            x + s**3 if want_gradient else None,
            np.eye(2) + 3 * s * s if hessian_kind is not None else None,
        )


def main():
    fixed, dynamic = Quadratic(), Coupled()
    x = np.ones(2)
    value, g, H = fixed.eval(x, gradient=True, hessian=energy.HessianKind.Exact)
    active_value = dynamic.eval_value(x)  # Value-only requests need no pattern update.
    dynamic.update_pattern(x)
    _, g, H = dynamic.eval(x, gradient=True, hessian=energy.HessianKind.Exact)
    active_nnz = H.nnz
    dynamic.update_pattern(-x)
    H = dynamic.eval_hessian(-x)
    print(
        f"fixed value={value:g}, active value={active_value:g}, Hessian nnz=2->{active_nnz}->{H.nnz}"
    )


if __name__ == "__main__":
    main()
