"""Differentiate equilibrium with Torch material/Fp force responses.

Run: uv run --no-sync python examples/fem/fitting/torch_equilibrium_response.py
The adjacent torch_neohookean.py supplies the density and two-tet problem.
"""

from litepgo.energy import AffineResponse
import numpy as np
from litepgo import dof, fem, mesh, solver
from litepgo import torch as bridge
from torch_neohookean import make_problem


def make_equilibrium(kind, use_torch=True):
    _, mapping, geometry, plastic, theta, _, torch_energy = make_problem()
    topology = mesh.TetTopology(
        5, np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32)
    )
    if kind == "material":
        parameters = np.array([2.0, 3.0])
        fixed_fields = plastic
        TorchResponse = bridge.TorchVolumetricFemMaterialResponse
        NativeResponse = fem.VolumetricFemForceMaterialResponse[
            fem.NeoHookean, fem.TetP1Basis, fem.TetCentroidQuadrature
        ]
    elif kind == "fp":
        parameters = plastic[0].eval_value(np.full(3, 0.25))
        fixed_fields = theta
        TorchResponse = bridge.TorchVolumetricFemFpResponse
        NativeResponse = fem.VolumetricFemForceFpResponse[
            fem.NeoHookean, fem.TetP1Basis, fem.TetCentroidQuadrature
        ]
    else:
        raise ValueError("kind must be material or fp")
    parameter_map = dof.ConstantDofMap(topology, len(parameters))
    NativeEnergy = fem.VolumetricFemEnergy[
        fem.NeoHookean, fem.TetP1Basis, fem.TetCentroidQuadrature
    ]

    def make_energy(p):
        active = [AffineResponse(np.zeros((len(p), 3)), p)] * mapping.num_elements
        Fp, material_fields = (
            (plastic, active) if kind == "material" else (active, theta)
        )
        if use_torch:
            return bridge.TorchVolumetricFemEnergy(
                mapping, geometry, Fp, material_fields, material=torch_energy.material
            )
        return NativeEnergy(mapping, geometry, Fp, material_fields)

    def make_response(u):
        # Same density and field convention as make_energy: response(p) = dE(u,p)/du.
        args = (parameter_map, mapping, geometry, u, fixed_fields)
        return (
            TorchResponse(*args, material=torch_energy.material)
            if use_torch
            else NativeResponse(*args)
        )

    fixed = dof.DofReduction(15, [*range(9), 12, 13, 14])
    initial = np.zeros(15)
    initial[12:] = [0.04, -0.02, 0.06]
    equilibrium = solver.StaticEquilibriumResponse(
        len(parameters),
        fixed,
        initial,
        solver.NewtonOptions(gradient_tolerance=1e-12, max_iterations=100),
        make_energy,
        make_response,
    )
    return equilibrium, parameters


def main():
    for kind in ("material", "fp"):
        equilibrium, p = make_equilibrium(kind)
        native, _ = make_equilibrium(kind, use_torch=False)
        u, J = equilibrium.eval_value_and_jacobian(p)
        expected_u, expected_J = native.eval_value_and_jacobian(p)
        v = np.linspace(0.1, 0.3, len(p))
        w = np.linspace(-0.2, 0.4, len(u))
        # These calls solve directional/adjoint equations and use the force products.
        jv = equilibrium.eval_jvp(p, v)
        jtw = equilibrium.eval_vjp(p, w)
        print(
            f"{kind}: state_error={np.linalg.norm(u - expected_u):.3e}, "
            f"Jacobian_error={np.linalg.norm((J - expected_J).toarray()):.3e}, "
            f"JVP_error={np.linalg.norm(jv - expected_J @ v):.3e}, "
            f"VJP_error={np.linalg.norm(jtw - expected_J.T @ w):.3e}"
        )


if __name__ == "__main__":
    main()
