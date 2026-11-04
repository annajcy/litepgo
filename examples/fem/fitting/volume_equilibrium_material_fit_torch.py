"""Fit two material parameters from equilibrium reactions with Torch L-BFGS.

Run with the optional torch dependency installed. The FEM problem includes
both displacement and direct material contributions to reaction derivatives.

Setup: uv pip install torch
Run: uv run --no-sync python examples/fem/fitting/volume_equilibrium_material_fit_torch.py
Requires a built litepgo Python package. Uses CPU float64 and Torch L-BFGS.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from litepgo import torch as lp_torch
from volume_equilibrium_material_fit import make_problem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, help="rest tetrahedral VEG mesh")
    args = parser.parse_args()
    residual, truth, reaction, solve_count = make_problem(args.input)
    log_parameters = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    # L-BFGS may evaluate this closure repeatedly during its line search.
    optimizer = torch.optim.LBFGS(
        [log_parameters],
        max_iter=60,
        tolerance_grad=1e-10,
        tolerance_change=1e-14,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        r = lp_torch.response(residual, log_parameters)
        loss = 0.5 * r.square().sum()
        # The adapter applies the FEM Jacobian transpose to the upstream gradient.
        loss.backward()
        return loss

    optimizer.step(closure)
    fitted = log_parameters.detach().exp().numpy()
    error = np.linalg.norm(fitted - truth) / np.linalg.norm(truth)
    # Validate on a load that was not used to construct the fitting target.
    reference = reaction(truth, 2, -0.07, False)[0]
    predicted = reaction(fitted, 2, -0.07, False)[0]
    held_out = abs(predicted - reference) / abs(reference)
    print(
        f"truth={truth}, fitted={fitted}, parameter error={error:.3g}, "
        f"held-out error={held_out:.3g}, solves={solve_count()}"
    )
    return 0 if error < 1e-4 and held_out < 1e-4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
