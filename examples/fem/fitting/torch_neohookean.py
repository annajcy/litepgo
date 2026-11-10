"""Write W in PyTorch and compare native FEM assembly with native Neo-Hookean.

Run: uv run --no-sync python examples/fem/fitting/torch_neohookean.py
"""
from litepgo.solver import AbsoluteConvergence

from pathlib import Path

from litepgo.energy import AffineResponse
import numpy as np
import torch
from litepgo import dof, energy, fem, mesh, solver
from litepgo.torch import TorchVolumetricMaterial

REPO = Path(__file__).resolve().parents[3]


class TorchNeoHookean(TorchVolumetricMaterial):
    """The native NeoHookean density written in Torch; theta = [lambda, mu]."""

    num_parameters = 2

    @staticmethod
    def W(F, Fp, theta):
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


def make_problem(batch_size=256):
    """Two TetP1 elements with per-element Fp and theta fields."""
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
    key = (fem.TetP1Basis, fem.TetCentroidQuadrature)
    native = fem.VolumetricFemEnergy[key](fem.PointwiseVolumetricAdapter[fem.NeoHookean](), mapping, geometry, plastic, theta)
    # The same FEM type takes either native or Torch batch material instances.
    autodiff = fem.VolumetricFemEnergy[key](TorchNeoHookean(),
        mapping, geometry, plastic, theta, batch_size
    )
    return nodes, mapping, geometry, plastic, theta, native, autodiff


def compare(native, autodiff, u, label):
    for kind in (False, True):
        native_evaluation = native.evaluate(u)
        native_evaluation_hessian = native_evaluation.projected_hessian() if kind else native_evaluation.hessian()
        native_evaluation_gradient = native_evaluation.gradient()
        reference = (native_evaluation.value, native_evaluation_gradient, native_evaluation_hessian)
        autodiff_evaluation = autodiff.evaluate(u)
        autodiff_evaluation_hessian = autodiff_evaluation.projected_hessian() if kind else autodiff_evaluation.hessian()
        autodiff_evaluation_gradient = autodiff_evaluation.gradient()
        actual = (autodiff_evaluation.value, autodiff_evaluation_gradient, autodiff_evaluation_hessian)
        print(
            f"{label} {'projected' if kind else 'exact'}: energy_error={abs(actual[0] - reference[0]):.3e}, "
            f"force_error={np.linalg.norm(actual[1] - reference[1]):.3e}, "
            f"stiffness_error={np.linalg.norm((actual[2] - reference[2]).toarray()):.3e}"
        )


def hex_box():
    """The bundled box mesh with HexQ1 displacement and 2x2x2 Gauss points."""
    rest = mesh.read_veg(REPO / "assets/box-hex.veg")
    topology = rest.topology
    mapping = dof.VertexDofMap(topology, 3)
    XField = fem.InterpolatedResponse[fem.HexQ1Basis]
    geometry = [XField(mapping.gather(e, rest.positions.ravel())) for e in range(topology.num_cells)]
    Fp = [AffineResponse(np.zeros((9, 3)), np.eye(3).ravel(order="F"))] * topology.num_cells
    theta = [AffineResponse(np.zeros((2, 3)), np.array([2.0, 3.0]))] * topology.num_cells
    key = (fem.HexQ1Basis, fem.GaussLegendre3D[2])
    native = fem.VolumetricFemEnergy[key](fem.PointwiseVolumetricAdapter[fem.NeoHookean](), mapping, geometry, Fp, theta)
    autodiff = fem.VolumetricFemEnergy[key](TorchNeoHookean(), mapping, geometry, Fp, theta)
    stretch = np.diag([1.06, 0.97, 1.02]) + 0.02 * np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    u = (rest.positions @ (stretch - np.eye(3)).T).ravel()
    return native, autodiff, u


def main():
    nodes, _, _, _, _, native, autodiff = make_problem()
    F = np.array([[1.08, 0.04, 0.02], [0.01, 0.96, 0.03], [0.0, 0.02, 1.05]])
    u = (nodes @ (F - np.eye(3)).T).ravel()
    compare(native, autodiff, u, "TetP1")
    hex_native, hex_autodiff, hex_u = hex_box()
    compare(hex_native, hex_autodiff, hex_u, "HexQ1")

    fixed = dof.DofReduction(15, [*range(9), 12, 13, 14])
    initial = np.zeros(15)
    initial[12:] = [0.04, -0.02, 0.06]
    options = solver.NewtonOptions(max_iterations=100)
    reference = solver.newton_solve(native, fixed, initial, options, convergence=AbsoluteConvergence(1e-10))
    actual = solver.newton_solve(autodiff, fixed, initial, options, convergence=AbsoluteConvergence(1e-10))
    if any(
        result.status != solver.NewtonStatus.Converged for result in (reference, actual)
    ):
        raise RuntimeError("Equilibrium solve did not converge")
    print(f"equilibrium_error={np.linalg.norm(actual.x - reference.x):.3e}")


if __name__ == "__main__":
    main()
