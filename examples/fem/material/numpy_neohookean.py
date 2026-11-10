"""A Neo-Hookean material written in NumPy, assembled by native Hermite-hex FEM.

The material implements the full PyVolumetricMaterial contract with closed-form,
vectorized derivatives, so it also drives both force responses. The script checks
it against the native NeoHookean and solves a corner-pull equilibrium with each.
Run: uv run --no-sync python examples/fem/material/numpy_neohookean.py
"""

import argparse
from pathlib import Path

import numpy as np
from litepgo import dof, energy, fem, mesh, solver
from litepgo.energy import AffineResponse

REPO = Path(__file__).resolve().parents[3]


class NumpyNeoHookean(fem.PyVolumetricMaterial):
    """W = det(Fp) psi(Fe), Fe = F Fp^-1, theta = [lambda, mu], as in fem.NeoHookean.

    psi = mu/2 (|Fe|^2 - 3) - mu log J + lambda/2 log(J)^2 with J = det(Fe).
    Outside det(Fe) > 0 the logarithm is NaN, which rejects the evaluation.
    """

    num_parameters = 2

    @staticmethod
    def eval(F, Fp, theta, *, outputs):
        G = np.linalg.inv(Fp)
        GT = G.transpose(0, 2, 1)
        d = np.linalg.det(Fp)
        Fe = F @ G
        with np.errstate(invalid="ignore", divide="ignore"):
            log_J = np.log(np.linalg.det(Fe))
        lam, mu = theta[:, 0], theta[:, 1]
        Finv = np.linalg.inv(Fe)
        FinvT = Finv.transpose(0, 2, 1)
        c = lam * log_J - mu
        # S = dpsi/dFe; P = dW/dF = det(Fp) S Fp^-T.
        S = mu[:, None, None] * Fe + c[:, None, None] * FinvT
        P = d[:, None, None] * (S @ GT)
        result = {}
        if "W" in outputs:
            psi = 0.5 * mu * ((Fe**2).sum((1, 2)) - 3) - mu * log_J + 0.5 * lam * log_J**2
            result["W"] = d * psi
        if "dW_dF" in outputs:
            result["dW_dF"] = P
        if "d2W_dF2" in outputs or "d2W_dF_dFp" in outputs:
            # C[a,b,m,n] = d2psi / dFe_ab dFe_mn.
            identity = np.einsum("am,bn->abmn", np.eye(3), np.eye(3))
            C = (
                mu[:, None, None, None, None] * identity
                + lam[:, None, None, None, None] * np.einsum("zba,znm->zabmn", Finv, Finv)
                - c[:, None, None, None, None] * np.einsum("zbm,zna->zabmn", Finv, Finv)
            )
        if "d2W_dF2" in outputs:
            # dFe_kn/dF_kl = G_ln, so A[i,j,k,l] = det(Fp) C[i,b,k,n] G[l,n] G[j,b].
            result["d2W_dF2"] = d[:, None, None, None, None] * np.einsum(
                "zibkn,zln,zjb->zijkl", C, G, G, optimize=True
            )
        if "d2W_dF_dtheta" in outputs:
            dS = np.stack([log_J[:, None, None] * FinvT, Fe - FinvT], axis=-1)
            result["d2W_dF_dtheta"] = d[:, None, None, None] * np.einsum("zibt,zjb->zijt", dS, G)
        if "d2W_dF_dFp" in outputs:
            # d det(Fp)/dFp_kl = det(Fp) G_lk, dG_jb/dFp_kl = -G_jk G_lb and
            # dFe_mn/dFp_kl = -Fe_mk G_ln.
            scale = d[:, None, None, None, None]
            result["d2W_dF_dFp"] = scale * (
                np.einsum("zij,zlk->zijkl", S @ GT, G)
                - np.einsum("zibmn,zmk,zln,zjb->zijkl", C, Fe, G, G, optimize=True)
                - np.einsum("zib,zjk,zlb->zijkl", S, G, G)
            )
        return result


# 4x4x4 Gauss points: 2x2x2 under-integrates Hermite hexes and admits zero-energy modes.
KEY = (fem.HexHermiteBasis, fem.GaussLegendre3D[4])


def make_problem(input_path):
    rest = mesh.read_veg(input_path)
    topology = rest.topology
    rest_map = dof.VertexDofMap(topology, 3)
    mapping = dof.VertexDofMap(topology, 24)  # Hermite: 3 displacements x 8 per vertex.
    XField = fem.InterpolatedResponse[fem.HexQ1Basis]
    geometry = [XField(rest_map.gather(e, rest.positions.ravel())) for e in range(topology.num_cells)]
    plastic = np.array([[1.05, 0.02, 0.0], [0.0, 0.97, 0.03], [0.01, 0.0, 1.02]])
    Fp = [AffineResponse(np.zeros((9, 3)), plastic.ravel(order="F"))] * topology.num_cells
    theta = [AffineResponse(np.zeros((2, 3)), np.array([2.0, 3.0]))] * topology.num_cells
    return rest, mapping, geometry, Fp, theta


def corner_pull(elastic, rest):
    nodes = rest.positions
    lower, upper = nodes.min(axis=0), nodes.max(axis=0)
    corners = np.flatnonzero(
        ((np.abs(nodes - lower) < 1e-9) | (np.abs(nodes - upper) < 1e-9)).all(axis=1)
    )
    pulled = (24 * corners[:, None] + np.arange(3)).astype(np.int32)
    pulling = energy.VertexPullingEnergy(elastic.num_dofs, pulled, 0.3 * (nodes[corners] - lower), 1.0)
    potential = energy.WeightedSumEnergy([(elastic, 1.0), (pulling, 1.0)])
    options = solver.NewtonOptions(gradient_tolerance=1e-11)
    result = solver.newton_solve(
        potential, dof.DofReduction(elastic.num_dofs, []), np.zeros(elastic.num_dofs), options
    )
    if result.status != solver.NewtonStatus.Converged:
        raise RuntimeError("Corner-pull equilibrium did not converge")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=REPO / "assets/box-hex.veg", help="rest hexahedral VEG mesh")
    rest, mapping, geometry, Fp, theta = make_problem(parser.parse_args().input)
    native = fem.VolumetricFemEnergy[KEY](fem.PointwiseVolumetricAdapter[fem.NeoHookean](), mapping, geometry, Fp, theta)
    numpy = fem.VolumetricFemEnergy[KEY](NumpyNeoHookean(), mapping, geometry, Fp, theta)
    u = 0.02 * np.sin(np.arange(mapping.num_dofs) * 0.37)
    native_evaluation = native.evaluate(u)
    native_evaluation_hessian = native_evaluation.hessian()
    native_evaluation_gradient = native_evaluation.gradient()
    a = (native_evaluation.value, native_evaluation_gradient, native_evaluation_hessian)
    numpy_evaluation = numpy.evaluate(u)
    numpy_evaluation_hessian = numpy_evaluation.hessian()
    numpy_evaluation_gradient = numpy_evaluation.gradient()
    b = (numpy_evaluation.value, numpy_evaluation_gradient, numpy_evaluation_hessian)
    print(
        f"energy: value_error={abs(a[0] - b[0]):.2e}, "
        f"gradient_error={np.abs(a[1] - b[1]).max():.2e}, "
        f"hessian_error={abs(a[2] - b[2]).max():.2e}"
    )

    topology = rest.topology
    for Force, fixed, p in (
        (fem.VolumetricFemForceMaterialResponse, Fp, np.array([2.5, 1.5])),
        (fem.VolumetricFemForceFpResponse, theta, np.diag([1.1, 0.95, 1.0]).ravel(order="F")),
    ):
        parameter_map = dof.ConstantDofMap(topology, len(p))
        reference = Force[KEY](fem.PointwiseVolumetricAdapter[fem.NeoHookean](), parameter_map, mapping, geometry, u, fixed)
        actual = Force[KEY](NumpyNeoHookean(), parameter_map, mapping, geometry, u, fixed)
        print(f'{Force.__name__}: value_error={np.abs(actual.evaluate(p).value - reference.evaluate(p).value).max():.2e}, jacobian_error={abs(actual.evaluate(p).jacobian() - reference.evaluate(p).jacobian()).max():.2e}')

    reference = corner_pull(native, rest)
    actual = corner_pull(numpy, rest)
    print(
        f"corner pull: iterations numpy={actual.iterations} native={reference.iterations}, "
        # Hermite derivative DOFs are large here; parallel sums vary by roundoff run to run.
        f"relative_displacement_error={np.abs(actual.x - reference.x).max() / np.abs(reference.x).max():.2e}"
    )


if __name__ == "__main__":
    main()
