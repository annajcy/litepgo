"""Fit shell plastic stretches with Torch L-BFGS.

The native example beside this file defines the physical model and observations.
"""

from functools import partial
import numpy as np
import torch
from litepgo import torch as lp_torch
from shell_equilibrium_fp_fit import make_problem, make_factories, measure, Residual


def main():
    problem = make_problem()
    make_energy, make_response = make_factories(problem)
    measurement = partial(measure, problem, make_energy, make_response)
    truth = np.array([1.06, 0.95])
    target, _ = measurement(np.log(truth))
    num_free = len(problem.fixed.free_dofs)
    scales = np.concatenate(
        [
            np.full(num_free, max(np.linalg.norm(target[:num_free]), 1e-8)),
            np.full(
                len(target) - num_free, max(np.linalg.norm(target[num_free:]), 1e-8)
            ),
        ]
    )
    residual = Residual(measurement, target, scales)

    parameters = torch.zeros(len(truth), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [parameters],
        max_iter=150,
        tolerance_grad=1e-10,
        tolerance_change=1e-24,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        loss = 0.5 * lp_torch.response(residual, parameters).square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    closure()
    converged = parameters.grad.abs().max().item() <= 1e-10
    fitted_parameters = parameters.detach().numpy().copy()
    fitted = np.exp(fitted_parameters)
    parameter_error = np.linalg.norm(fitted - truth) / np.linalg.norm(truth)
    relative_residual = np.linalg.norm(residual.evaluate(fitted_parameters).value)
    held_out, _ = measurement(np.log(truth), load=0.6)
    prediction, _ = measurement(fitted_parameters, load=0.6)
    held_out_error = np.linalg.norm((prediction - held_out) / scales)
    print(
        "truth="
        + " ".join(f"{x:.12g}" for x in truth)
        + ", fitted="
        + " ".join(f"{x:.12g}" for x in fitted)
        + f", converged={int(converged)}, parameter_error={parameter_error:.12g}"
        + f", relative_residual={relative_residual:.12g}, held_out_error={held_out_error:.12g}"
    )
    return 0 if converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
