"""Differentiate equilibrium with force responses of a Torch material.

Run: uv run --no-sync python examples/fem/fitting/torch_equilibrium_response.py
The adjacent torch_neohookean.py supplies the density and two-tet problem.
"""
from litepgo.solver import AbsoluteConvergence

from litepgo.energy import AffineResponse
import numpy as np
from litepgo import dof, fem, mesh, solver
from torch_neohookean import TorchNeoHookean, make_problem


def make_equilibrium(kind, use_torch=True):
    _, mapping, geometry, plastic, theta, _, _ = make_problem()
    material = TorchNeoHookean() if use_torch else fem.PointwiseVolumetricAdapter[fem.NeoHookean]()
    key = (fem.TetP1Basis, fem.TetCentroidQuadrature)
    topology = mesh.TetTopology(
        5, np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32)
    )
    if kind == "material":
        parameters = np.array([2.0, 3.0])
        fixed_fields = plastic
        Response = fem.VolumetricFemForceMaterialResponse[key]
    elif kind == "fp":
        parameters = plastic[0].evaluate(np.full(3, 0.25)).value
        fixed_fields = theta
        Response = fem.VolumetricFemForceFpResponse[key]
    else:
        raise ValueError("kind must be material or fp")
    parameter_map = dof.ConstantDofMap(topology, len(parameters))
    Energy = fem.VolumetricFemEnergy[key]

    def make_energy(p):
        active = [AffineResponse(np.zeros((len(p), 3)), p)] * mapping.num_elements
        Fp, material_fields = (
            (plastic, active) if kind == "material" else (active, theta)
        )
        return Energy(material, mapping, geometry, Fp, material_fields)

    def make_response(u):
        # Same density and field convention as make_energy: response(p) = dE(u,p)/du.
        return Response(material, parameter_map, mapping, geometry, u, fixed_fields)

    fixed = dof.DofReduction(15, [*range(9), 12, 13, 14])
    initial = np.zeros(15)
    initial[12:] = [0.04, -0.02, 0.06]
    equilibrium = solver.StaticEquilibriumResponse(
        len(parameters),
        fixed,
        initial,
        solver.NewtonOptions(max_iterations=100),
        make_energy,
        make_response,
        convergence=AbsoluteConvergence(1e-12),
    )
    return equilibrium, parameters


def main():
    for kind in ("material", "fp"):
        equilibrium, p = make_equilibrium(kind)
        native, _ = make_equilibrium(kind, use_torch=False)
        equilibrium_evaluation = equilibrium.evaluate(p)
        J = equilibrium_evaluation.jacobian()
        u = equilibrium_evaluation.value
        native_evaluation = native.evaluate(p)
        expected_J = native_evaluation.jacobian()
        expected_u = native_evaluation.value
        v = np.linspace(0.1, 0.3, len(p))
        w = np.linspace(-0.2, 0.4, len(u))
        # These calls solve directional/adjoint equations and use the force products.
        jv = equilibrium.evaluate(p).jvp(v)
        jtw = equilibrium.evaluate(p).vjp(w)
        print(
            f"{kind}: state_error={np.linalg.norm(u - expected_u):.3e}, "
            f"Jacobian_error={np.linalg.norm((J - expected_J).toarray()):.3e}, "
            f"JVP_error={np.linalg.norm(jv - expected_J @ v):.3e}, "
            f"VJP_error={np.linalg.norm(jtw - expected_J.T @ w):.3e}"
        )


if __name__ == "__main__":
    main()
