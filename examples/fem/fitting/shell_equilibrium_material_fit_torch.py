"""Fit shell material with L-BFGS using the model from the native example."""

from functools import partial
import numpy as np
import torch
from litepgo import torch as lp_torch
from shell_equilibrium_material_fit import (
    make_problem,
    make_factories,
    measure,
    Residual,
)


def main():
    problem = make_problem()
    make_energy, make_response = make_factories(problem)
    measurement = partial(measure, problem, make_energy, make_response)
    truth, initial = np.array([2.0, 3.0]), np.array([0.7, 1.2])
    target = measurement(truth).reaction
    scale = np.linalg.norm(target)
    residual = Residual(measurement, target, scale)
    optimization_parameters = torch.tensor(
        initial, dtype=torch.float64, requires_grad=True
    )
    # L-BFGS may evaluate this closure repeatedly during its line search.
    optimizer = torch.optim.LBFGS(
        [optimization_parameters],
        max_iter=150,
        tolerance_grad=1e-10,
        tolerance_change=1e-24,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        loss = 0.5 * lp_torch.response(residual, optimization_parameters).square().sum()
        # The adapter applies the FEM Jacobian transpose to the upstream gradient.
        loss.backward()
        return loss

    optimizer.step(closure)
    closure()
    converged = optimization_parameters.grad.abs().max().item() <= 1e-10
    fitted_parameters = optimization_parameters.detach().numpy().copy()
    parameter_error = np.linalg.norm(fitted_parameters - truth) / np.linalg.norm(truth)
    relative_residual = (
        np.linalg.norm(measurement(fitted_parameters).reaction - target) / scale
    )
    # Validate on a load that was not used to construct the fitting target.
    held_out = measurement(truth, load=0.6).reaction
    held_out_error = np.linalg.norm(
        measurement(fitted_parameters, load=0.6).reaction - held_out
    ) / np.linalg.norm(held_out)
    print(
        f"fitted={fitted_parameters[0]:.12g} {fitted_parameters[1]:.12g}, converged={int(converged)}, "
        f"parameter_error={parameter_error:.12g}, relative_residual={relative_residual:.12g}, "
        f"held_out_error={held_out_error:.12g}"
    )
    return 0 if converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
