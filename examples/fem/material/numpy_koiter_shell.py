"""A Koiter shell material written in NumPy, assembled by native mid-edge shell FEM.

Inputs are packed forms (a11,a12,a22,b11,b12,b22) with independent mixed entries,
so each density term is a quadratic form in packed vectors. The script compares
the material with the native KoiterStVK and solves a clamped-patch equilibrium.
Run: uv run --no-sync python examples/fem/material/numpy_koiter_shell.py
"""

import numpy as np
from litepgo import dof, energy, fem, mesh, solver

# Symmetric 2x2 basis matrices for the packed entries 11, 12 (both off-diagonals), 22.
BASIS = np.array([[[1.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]])


class NumpyKoiter(fem.PyShellMaterial):
    """W = t psi((a - a_bar)/2) + t^3/12 psi(b - b_bar), theta = [lambda_ps, mu].

    psi(S) = lambda/2 tr(A^-1 S)^2 + mu tr(A^-1 S A^-1 S) with A = a_bar. For packed
    s, psi = s^T Q s / 2 where Q = lambda l l^T + 2 mu T, l_r = tr(A^-1 E_r) and
    T_rs = tr(A^-1 E_r A^-1 E_s).
    """

    num_parameters = 2

    @staticmethod
    def eval(ab, ab_bar, thickness, theta, *, outputs):
        p = ab_bar[:, :3]
        inverse = np.stack([p[:, 2], -p[:, 1], -p[:, 1], p[:, 0]], -1).reshape(-1, 2, 2)
        inverse /= (p[:, 0] * p[:, 2] - p[:, 1] ** 2)[:, None, None]
        AE = np.einsum("zij,rjk->zrik", inverse, BASIS)  # A^-1 E_r
        l = np.einsum("zrii->zr", AE)
        T = np.einsum("zrij,zsji->zrs", AE, AE)
        lam, mu = theta[:, 0, None, None], theta[:, 1, None, None]
        Q_lam, Q_mu = np.einsum("zr,zs->zrs", l, l), 2 * T
        Q = lam * Q_lam + mu * Q_mu
        e = 0.5 * (ab[:, :3] - ab_bar[:, :3])
        k = ab[:, 3:] - ab_bar[:, 3:]
        t = thickness[:, None]
        bend = t**3 / 12
        result = {}
        if "W" in outputs:
            result["W"] = 0.5 * (
                thickness * np.einsum("zr,zrs,zs->z", e, Q, e)
                + bend[:, 0] * np.einsum("zr,zrs,zs->z", k, Q, k)
            )
        if "dW_dab" in outputs:
            result["dW_dab"] = np.concatenate(
                [0.5 * t * np.einsum("zrs,zs->zr", Q, e), bend * np.einsum("zrs,zs->zr", Q, k)], 1
            )
        if "d2W_dab2" in outputs:
            H = np.zeros((len(ab), 6, 6))
            H[:, :3, :3] = 0.25 * t[:, :, None] * Q
            H[:, 3:, 3:] = bend[:, :, None] * Q
            result["d2W_dab2"] = H
        if "d2W_dab_dtheta" in outputs:
            columns = [
                np.concatenate([0.5 * t * np.einsum("zrs,zs->zr", D, e), bend * np.einsum("zrs,zs->zr", D, k)], 1)
                for D in (Q_lam, Q_mu)
            ]
            result["d2W_dab_dtheta"] = np.stack(columns, -1)
        if "d2W_dab_dabarbbar" in outputs:
            # dA^-1/da_bar_c = -A^-1 E_c A^-1, so dl_r = -T_cr and
            # dT_rs = -(R_crs + R_rcs) with R_xyz = tr(A^-1 E_x A^-1 E_y A^-1 E_z).
            R = np.einsum("zxij,zyjk,zwki->zxyw", AE, AE, AE)
            dQ_lam = -(np.einsum("zcr,zs->zcrs", T, l) + np.einsum("zr,zcs->zcrs", l, T))
            dQ_mu = -2 * (R + R.transpose(0, 2, 1, 3))
            dQ = lam[..., None] * dQ_lam + mu[..., None] * dQ_mu  # [z, c, r, s]
            M = np.zeros((len(ab), 6, 6))
            M[:, :3, :3] = 0.5 * t[:, :, None] * (np.einsum("zcrs,zs->zrc", dQ, e) - 0.5 * Q)
            M[:, 3:, :3] = bend[:, :, None] * np.einsum("zcrs,zs->zrc", dQ, k)
            M[:, 3:, 3:] = -bend[:, :, None] * Q
            result["d2W_dab_dabarbbar"] = M
        return result


def make_problem():
    """A slightly curved 3x3-node patch; boundary nodes are prescribed, the center is free."""
    nodes = np.array([[x, y, 0.05 * x * y] for y in (-1.0, 0.0, 1.0) for x in (-1.0, 0.0, 1.0)])
    faces = []
    for y in range(2):
        for x in range(2):
            a = 3 * y + x
            faces.extend([[a, a + 1, a + 4], [a, a + 4, a + 3]])
    topology = mesh.TriangleTopology(9, np.array(faces, dtype=np.int32))
    stencils = mesh.make_triangle_mid_edge_stencils(topology)
    mapping = dof.StencilDofMap(9, stencils.vertices, 3)
    rest = [
        fem.TriangleMidEdgeRestData(mapping.gather(e, nodes.ravel()), stencils.opposite[e])
        for e in range(mapping.num_elements)
    ]
    count = mapping.num_elements
    thickness = [0.1] * count
    theta = [np.array([2.0, 3.0])] * count
    Fp = [np.diag([1.04, 0.98, 1.0]) + 0.01 * e * np.eye(3) for e in range(count)]
    X, Y = nodes[:, 0], nodes[:, 1]
    prescribed = np.column_stack([0.08 * X + 0.04 * Y, 0.03 * Y + 0.02 * X * Y, 0.08 * X * Y]).ravel()
    return topology, mapping, rest, thickness, theta, Fp, prescribed


def main():
    topology, mapping, rest, thickness, theta, Fp, prescribed = make_problem()
    native = fem.ShellFemEnergy[fem.KoiterStVK](mapping, rest, thickness, theta, Fp)
    numpy = fem.ShellFemEnergy[NumpyKoiter](mapping, rest, thickness, theta, Fp)
    u = prescribed + 0.02 * np.sin(np.arange(27))
    a = native.eval(u, gradient=True, hessian=energy.HessianKind.Exact)
    b = numpy.eval(u, gradient=True, hessian=energy.HessianKind.Exact)
    print(
        f"energy: value_error={abs(a[0] - b[0]):.2e}, "
        f"gradient_error={np.abs(a[1] - b[1]).max():.2e}, "
        f"hessian_error={abs(a[2] - b[2]).max():.2e}"
    )

    material = (fem.ShellFemForceMaterialResponse, (u, thickness), np.array([2.5, 1.5]))
    plastic = (fem.ShellFemForceFpResponse, (u, thickness, theta), np.diag([1.1, 0.95, 1.0]).ravel(order="F"))
    for Force, fixed, p in (material, plastic):
        parameter_map = dof.ConstantDofMap(topology, len(p))
        reference = Force[fem.KoiterStVK](parameter_map, mapping, rest, *fixed)
        actual = Force[NumpyKoiter](parameter_map, mapping, rest, *fixed)
        print(
            f"{Force.__name__}: value_error={np.abs(actual.eval_value(p) - reference.eval_value(p)).max():.2e}, "
            f"jacobian_error={abs(actual.eval_jacobian(p) - reference.eval_jacobian(p)).max():.2e}"
        )

    fixed = dof.DofReduction(27, [i for i in range(27) if i // 3 != 4])
    options = solver.NewtonOptions(gradient_tolerance=1e-12, max_iterations=100)
    results = [solver.newton_solve(E, fixed, prescribed, options) for E in (native, numpy)]
    if any(r.status != solver.NewtonStatus.Converged for r in results):
        raise RuntimeError("Shell equilibrium did not converge")
    print(f"equilibrium: center_displacement_error={np.abs(results[0].x - results[1].x).max():.2e}")


if __name__ == "__main__":
    main()
