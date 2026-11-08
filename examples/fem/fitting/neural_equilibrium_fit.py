"""Fit two network weights through native FEM equilibrium (CPU float64).

Run: uv run --no-sync python examples/fem/fitting/neural_equilibrium_fit.py
Keep neural_neohookean.py and torch_neohookean.py beside this file.
"""

import json

import numpy as np
import scipy.sparse as sp
import torch

from litepgo import dof, energy, fem, mesh, solver, tbb_thread_limit
from litepgo.torch import TorchVolumetricMaterial, response
from neural_neohookean import InvariantMaterial
from torch_neohookean import TorchNeoHookean, make_problem


def make_material(num_parameters=2):
    if not isinstance(num_parameters, int) or not 1 <= num_parameters <= 32:
        raise ValueError("expected 1 to 32 active final-layer weights")
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = InvariantMaterial().eval().requires_grad_(False)
    fixed = dict(model.named_parameters())
    reference = fixed["net.4.weight"][0, :num_parameters].detach().clone()

    def unpack(p):
        if p.shape != (num_parameters,):
            raise ValueError(f"expected {num_parameters} final-layer weights")
        # Concatenation preserves derivatives to p without mutating the module.
        return {**fixed, "net.4.weight": torch.cat((p, fixed["net.4.weight"][0, num_parameters:]))[None]}

    def hidden(z):
        z = torch.nn.functional.softplus(torch.nn.functional.linear(z, fixed["net.0.weight"], fixed["net.0.bias"]))
        return torch.nn.functional.softplus(torch.nn.functional.linear(z, fixed["net.2.weight"], fixed["net.2.bias"]))

    # Hidden weights are fixed, so their reference features and slopes are constants.
    with torch.inference_mode(False), torch.enable_grad():
        origin = torch.zeros(3, dtype=torch.float64)
        h0 = hidden(origin).detach()
        slope = torch.autograd.functional.jacobian(hidden, origin)[:, 2].detach()
    width = num_parameters

    class WeightMaterial(TorchVolumetricMaterial):
        num_parameters = width

        @staticmethod
        def W(F, Fp, theta):
            physical = F.new_tensor([2.0, 3.0]).expand(len(F), 2)
            z = InvariantMaterial.invariants(F @ torch.linalg.inv(Fp))
            weights = torch.cat((theta, fixed["net.4.weight"][:, width:].expand(len(F), -1)), dim=1)
            features = hidden(z) - h0 - z[:, 2:3] * slope
            neural = torch.linalg.det(Fp) * (weights * features).sum(-1)
            return TorchNeoHookean.W(F, Fp, physical) + neural

    return WeightMaterial, model, unpack, reference


def make_equilibrium(material, batch_size=256):
    _, mapping, geometry, _, _, _, _ = make_problem()
    topology = mesh.TetTopology(5, np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32))
    plastic = [energy.AffineResponse(np.zeros((9, 3)), np.eye(3).ravel())] * 2
    parameter_map = dof.ConstantDofMap(topology, 2)
    key = (material, fem.TetP1Basis, fem.TetCentroidQuadrature)
    free = np.array([9, 10, 11])
    fixed = dof.DofReduction(15, [*range(9), 12, 13, 14])
    load = np.zeros(15)
    load[free] = [0.015, -0.01, 0.02]
    external = energy.QuadraticEnergy(sp.csc_matrix((15, 15)), -load)

    def make_energy(p):
        fields = [energy.AffineResponse(np.zeros((2, 3)), p)] * 2
        elastic = fem.VolumetricFemEnergy[key](mapping, geometry, plastic, fields, batch_size)
        return energy.WeightedSumEnergy([(elastic, 1.0), (external, 1.0)])

    def make_response(u):
        # The fixed external load has zero parameter derivative.
        return fem.VolumetricFemForceMaterialResponse[key](
            parameter_map, mapping, geometry, u, plastic, batch_size
        )

    equilibrium = solver.StaticEquilibriumResponse(
        2, fixed, np.zeros(15),
        solver.NewtonOptions(gradient_tolerance=1e-12, max_iterations=100),
        make_energy, make_response,
    )
    return equilibrium, free, make_energy, make_response


def make_fit(batch_size=256):
    material, _, _, reference = make_material()
    equilibrium, free, _, _ = make_equilibrium(material, batch_size)
    target = torch.from_numpy(equilibrium.eval_value(reference.numpy()).copy())
    initial = reference + reference.new_tensor([0.15, -0.12])
    initial_u = torch.from_numpy(equilibrium.eval_value(initial.numpy()).copy())
    # A fixed observation scale makes the small network-induced displacement visible.
    scale = (initial_u[free] - target[free]).square().mean().sqrt().item()
    if not np.isfinite(scale) or scale <= 1e-10:
        raise ValueError("selected weights do not measurably affect equilibrium")

    def loss(p):
        u = response(equilibrium, p)
        return 0.5 * ((u[free] - target[free]) / scale).square().mean()

    return equilibrium, initial, reference, loss


def fit(initial, loss):
    p = initial.detach().clone().requires_grad_()
    optimizer = torch.optim.LBFGS(
        [p], max_iter=20, tolerance_grad=1e-9, tolerance_change=1e-13,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        value = loss(p)
        value.backward()
        return value

    before = loss(p).item()
    optimizer.step(closure)
    return p.detach(), before, loss(p).item()


def main():
    torch.set_num_threads(1)
    with tbb_thread_limit(1):
        _, initial, reference, loss = make_fit()
        fitted, before, after = fit(initial, loss)
    print(json.dumps({
        "initial_loss": before, "final_loss": after,
        "initial_weights": initial.tolist(), "fitted_weights": fitted.tolist(),
        "reference_weights": reference.tolist(),
    }, indent=2))


if __name__ == "__main__":
    main()
