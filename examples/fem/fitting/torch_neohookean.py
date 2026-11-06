"""Write W in PyTorch and compare TetP1 FEM with native Neo-Hookean.

Run: uv run --no-sync python examples/fem/fitting/torch_neohookean.py
"""

from litepgo.energy import AffineResponse
import numpy as np
import torch
from litepgo import dof, energy, fem, mesh, solver
from litepgo.torch import TorchVolumetricFemEnergy


class NeoHookean(torch.nn.Module):
    def forward(self, F, Fp, theta):
        Fe = F @ torch.linalg.inv(Fp)
        log_J = torch.log(torch.linalg.det(Fe))
        lam, mu = theta.unbind(dim=-1)
        psi = (
            0.5 * mu * (Fe.square().sum(dim=(-2, -1)) - 3)
            - mu * log_J
            + 0.5 * lam * log_J.square()
        )
        # W is per original reference volume, matching the native material.
        W = torch.linalg.det(Fp) * psi
        # Match the native material domain; non-finite density rejects a trial.
        return torch.where((lam >= 0) & (mu > 0), W, torch.full_like(W, float("nan")))


def make_problem(batch_size=1):
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    topology = mesh.TetTopology(
        5, np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32)
    )
    mapping = dof.VertexDofMap(topology, 3)
    XField = fem.InterpolatedResponse[fem.TetP1Basis]
    geometry = [XField(mapping.gather(e, nodes.ravel())) for e in range(2)]
    plastic = [
        AffineResponse(np.zeros((9, 3)), p.ravel(order="F"))
        for p in (
            np.array([[1.04, 0.03, 0.0], [0.0, 0.98, 0.02], [0.01, 0.0, 1.02]]),
            np.array([[0.97, 0.0, 0.02], [0.01, 1.03, 0.0], [0.0, 0.02, 1.01]]),
        )
    ]
    theta = [AffineResponse(np.zeros((2, 3)), np.array(p)) for p in ([2.0, 3.0], [3.0, 2.0])]
    native = fem.VolumetricFemEnergy[
        fem.NeoHookean, fem.TetP1Basis, fem.TetCentroidQuadrature
    ](mapping, geometry, plastic, theta)
    autodiff = TorchVolumetricFemEnergy(
        mapping, geometry, plastic, theta, material=NeoHookean(), batch_size=batch_size
    )
    return nodes, mapping, geometry, plastic, theta, native, autodiff


def main():
    nodes, _, _, _, _, native, autodiff = make_problem()
    F = np.array([[1.08, 0.04, 0.02], [0.01, 0.96, 0.03], [0.0, 0.02, 1.05]])
    u = (nodes @ (F - np.eye(3)).T).ravel()
    for kind in (energy.HessianKind.Exact, energy.HessianKind.Projected):
        reference = native.eval(u, gradient=True, hessian=kind)
        actual = autodiff.eval(u, gradient=True, hessian=kind)
        print(
            f"{kind.name}: energy_error={abs(actual[0] - reference[0]):.3e}, "
            f"force_error={np.linalg.norm(actual[1] - reference[1]):.3e}, "
            f"stiffness_error={np.linalg.norm((actual[2] - reference[2]).toarray()):.3e}"
        )

    fixed = dof.DofReduction(15, [*range(9), 12, 13, 14])
    initial = np.zeros(15)
    initial[12:] = [0.04, -0.02, 0.06]
    options = solver.NewtonOptions(gradient_tolerance=1e-10, max_iterations=100)
    reference = solver.newton_solve(native, fixed, initial, options)
    actual = solver.newton_solve(autodiff, fixed, initial, options)
    if any(
        result.status != solver.NewtonStatus.Converged for result in (reference, actual)
    ):
        raise RuntimeError("Equilibrium solve did not converge")
    print(f"equilibrium_error={np.linalg.norm(actual.x - reference.x):.3e}")


if __name__ == "__main__":
    main()
