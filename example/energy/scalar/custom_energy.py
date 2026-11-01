"""Subclass fixed/dynamic Energy; request values, gradients and sparse Hessians.

Run from the repository root after building litepgo:
    .venv/bin/python example/energy/scalar/custom_energy.py
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
    np.testing.assert_allclose(g, [0.0, -1.0])
    assert value == 0.5
    assert dynamic.eval_value(x) == 5.0  # No pattern update needed for value only.
    assert dynamic.update_pattern(x)
    _, g, H = dynamic.eval(x, gradient=True, hessian=energy.HessianKind.Exact)
    np.testing.assert_allclose(g, [9.0, 9.0])
    assert H.nnz == 4 and H[0, 1] == 12
    assert not dynamic.update_pattern(x)
    assert dynamic.update_pattern(-x)
    H = dynamic.eval_hessian(-x)
    np.testing.assert_allclose(H.toarray(), np.eye(2))
    print(f"fixed value={value:g}, active value=5, Hessian nnz=2->4->{H.nnz}")


if __name__ == "__main__":
    main()
