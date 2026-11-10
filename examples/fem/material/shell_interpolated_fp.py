"""Sample plastic displacement gradients at shell face centroids, then relax."""

import json
from pathlib import Path
import numpy as np
from litepgo import dof, fem, mesh, solver, tbb_thread_limit


def plastic_fields(basis, lower, upper, affine=False):
    """Physical corner jets of one displacement, converted to basis coefficients."""
    size = upper - lower
    center = (upper + lower) / 2
    geometry = fem.InterpolatedResponse[fem.HexQ1Basis](
        np.array(
            [lower + size * fem.HexQ1Basis.coefficient_info(i).point for i in range(8)]
        )
    )
    rate = np.pi / 3 / size[2]  # 60 degrees across the box

    def sample(info):
        X = lower + size * np.asarray(info.point)
        dx, dy, dz = info.derivative_order
        value = np.zeros(3)
        if affine:
            if dx == dy == dz == 0:
                value[1] = -0.5 * (X[1] - center[1])
            elif (dx, dy, dz) == (0, 1, 0):
                value[1] = -0.5
            return value
        angle = rate * (X[2] - center[2])
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        J = np.array([[0.0, -1.0], [1.0, 0.0]])
        if dx and dy:
            return value
        v = (
            np.array([1.0, 0.0])
            if dx
            else np.array([0.0, 1.0])
            if dy
            else (X - center)[:2]
        )
        value[:2] = rate * J @ R @ v if dz else (R - np.eye(2)) @ v
        return value

    physical = np.array(
        [sample(basis.coefficient_info(i)) for i in range(basis.num_basis_functions)]
    )
    coefficients = fem.BasisDofTransform[basis](geometry).to_coefficients(
        physical.reshape(8, -1)
    )
    displacement = fem.InterpolatedResponse[basis](coefficients)
    return (
        geometry,
        displacement,
        fem.DeformationGradientResponse(displacement, geometry),
        physical,
    )


def analytic_fp(points, lower, upper):
    """Exact physical gradient of X + u_p_star for the prescribed twist."""
    center = (lower + upper) / 2
    rate = np.pi / 3 / (upper[2] - lower[2])
    r = points - center
    angle = rate * r[:, 2]
    c, s = np.cos(angle), np.sin(angle)
    F = np.zeros((len(points), 3, 3))
    F[:, 0, 0], F[:, 0, 1] = c, -s
    F[:, 1, 0], F[:, 1, 1] = s, c
    F[:, 0, 2] = rate * (-s * r[:, 0] - c * r[:, 1])
    F[:, 1, 2] = rate * (c * r[:, 0] - s * r[:, 1])
    F[:, 2, 2] = 1
    return F


def sample_fp(field, points):
    values = np.array(
        [field.evaluate(xi).value.reshape(3, 3, order="F") for xi in points]
    )
    assert np.isfinite(values).all() and np.all(np.linalg.det(values) > 0)
    return values


def check_fields():
    # Non-unit, translated box catches missing physical/reference derivative scaling.
    lower = np.array([-1.0, 2.0, 0.3])
    upper = lower + [2.0, 3.0, 4.0]
    points = np.random.default_rng(42).uniform(0.1, 0.9, (8, 3))
    for basis in [fem.HexQ1Basis, fem.HexHermiteBasis]:
        _, _, field, _ = plastic_fields(basis, lower, upper, affine=True)
        np.testing.assert_allclose(
            sample_fp(field, points),
            np.tile(np.diag([1.0, 0.5, 1.0]), (len(points), 1, 1)),
            atol=2e-14,
        )
        _, up, field, _ = plastic_fields(basis, lower, upper)
        for xi, Fp in zip(points, sample_fp(field, points)):
            h = 1e-6
            gradient = np.column_stack(
                [
                    (up.evaluate(xi + h * d).value - up.evaluate(xi - h * d).value)
                    / (2 * h)
                    for d in np.eye(3)
                ]
            )
            np.testing.assert_allclose(
                Fp,
                np.eye(3) + gradient / (upper - lower)[None, :],
                atol=2e-9,
                rtol=2e-9,
            )
    print(
        "Affine reproduction and physical-gradient finite differences passed",
        flush=True,
    )


def main():
    check_fields()
    root = Path(__file__).resolve().parents[3]
    out = root / "output/shell_interpolated_fp"
    out.mkdir(parents=True, exist_ok=True)
    original = mesh.read_obj(root / "assets/mesh/schwarz_p_open_cell.obj")
    X = np.array(original.positions)
    faces = np.array(original.cells)
    lower, upper = X.min(0), X.max(0)
    centroids = X[faces].mean(1)
    xi = (centroids - lower) / (upper - lower)
    stencils = mesh.make_triangle_mid_edge_stencils(original.topology)
    mapping = dof.StencilDofMap(len(X), stencils.vertices, 3)
    rest = [
        fem.TriangleMidEdgeRestData(mapping.gather(i, X.ravel()), stencils.opposite[i])
        for i in range(len(faces))
    ]
    theta = np.array([1e8 * 0.35 / (1 - 0.35**2), 1e8 / (2 * 1.35)])
    mesh.write_obj(out / "original.obj", original)
    report = {}
    for name, basis in [("q1", fem.HexQ1Basis), ("hermite", fem.HexHermiteBasis)]:
        geometry, up, field, physical = plastic_fields(basis, lower, upper)
        Fp = sample_fp(field, xi)
        fp_error = np.linalg.norm(
            Fp - analytic_fp(centroids, lower, upper), axis=(1, 2)
        )
        shell = fem.ShellFemEnergy(
            fem.PointwiseShellAdapter[fem.KoiterStVK](),
            mapping,
            rest,
            [0.06] * len(rest),
            [theta] * len(rest),
            Fp=Fp,
        )
        with tbb_thread_limit(4):
            initial = shell.evaluate(np.zeros(X.size)).value
            result = solver.newton_solve(
                shell,
                dof.DofReduction(X.size, []),
                np.zeros(X.size),
                solver.NewtonOptions(
                    max_iterations=1500, project_hessian=False, max_coordinate_step=0.01
                ),
                convergence=solver.AbsoluteConvergence(1e-5),
            )
        assert result.status == solver.NewtonStatus.Converged, result.status
        ev = shell.evaluate(result.x)
        residual = float(np.max(abs(ev.gradient())))
        assert residual < 1e-5 and ev.value < initial
        Y = X + result.x.reshape(-1, 3)
        # This sampled field shape is a visual reference, not a zero-energy oracle.
        target = X + np.array(
            [up.evaluate(point).value for point in (X - lower) / (upper - lower)]
        )
        mesh.write_obj(out / f"{name}_relaxed.obj", mesh.TriangleMesh(Y, faces))
        mesh.write_obj(
            out / f"{name}_field_target.obj", mesh.TriangleMesh(target, faces)
        )
        np.savez(
            out / f"{name}.npz",
            X=X,
            Y=Y,
            faces=faces,
            Fp=Fp,
            xi=xi,
            physical_dofs=physical,
            target=target,
        )
        report[name] = dict(
            mean_Fp_error=float(fp_error.mean()),
            max_Fp_error=float(fp_error.max()),
            iterations=result.iterations,
            initial_energy=initial,
            energy=ev.value,
            full_residual=residual,
            min_det_Fp=float(np.linalg.det(Fp).min()),
            max_det_Fp=float(np.linalg.det(Fp).max()),
        )
        print(name, report[name], flush=True)
        (out / "summary.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
