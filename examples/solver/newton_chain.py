"""Minimize 1/2 sum (x[i+1] - x[i])**2 with prescribed endpoints.

Run from the repository root after building litepgo:
    uv run --no-sync python examples/solver/newton_chain.py
"""
from litepgo.solver import AbsoluteConvergence

import numpy as np
import scipy.sparse as sp

from litepgo.dof import DofReduction
from litepgo.energy import QuadraticEnergy
from litepgo.solver import NewtonOptions, NewtonStatus, newton_solve


def main():
    n = 6
    differences = sp.diags(
        [-np.ones(n - 1), np.ones(n - 1)], [0, 1], shape=(n - 1, n), format="csc"
    )
    energy = QuadraticEnergy((differences.T @ differences).tocsc())
    initial = np.zeros(n)
    initial[0], initial[-1] = 1.0, 3.0
    # Fixed indices live in DofReduction; their values are taken from initial.
    fixed = DofReduction(n, [0, n - 1])

    def report(step):
        print(
            f"iteration={step.iteration}, energy={step.energy:g}, free_gradient={step.gradient_norm:g}, step={step.step:g}"
        )

    result = newton_solve(
        energy,
        fixed,
        initial,
        NewtonOptions(),
        on_iteration=report,
        convergence=AbsoluteConvergence(1e-10),
    )
    assert result.status == NewtonStatus.Converged
    np.testing.assert_array_equal(result.x[[0, -1]], initial[[0, -1]])
    np.testing.assert_allclose(result.x, np.linspace(1.0, 3.0, n), atol=1e-10)
    print(f"equilibrium={result.x}")


if __name__ == "__main__":
    main()
