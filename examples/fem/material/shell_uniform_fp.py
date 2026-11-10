"""Every face has Fp=diag(1,.5,1); relax directly from rest with Koiter-StVK."""

import json
from pathlib import Path
import numpy as np
from litepgo import dof, fem, mesh, solver, tbb_thread_limit


def main():
    root = Path(__file__).resolve().parents[3]
    out = root / "output/shell_uniform_fp"
    out.mkdir(parents=True, exist_ok=True)
    original = mesh.read_obj(root / "assets/mesh/schwarz_p_open_cell.obj")
    X, faces = np.array(original.positions), np.array(original.cells)
    stencils = mesh.make_triangle_mid_edge_stencils(original.topology)
    mapping = dof.StencilDofMap(len(X), stencils.vertices, 3)
    rest = [
        fem.TriangleMidEdgeRestData(mapping.gather(i, X.ravel()), stencils.opposite[i])
        for i in range(len(faces))
    ]
    theta = np.array([1e8 * 0.35 / (1 - 0.35**2), 1e8 / (2 * 1.35)])
    Fp = np.diag([1.0, 0.5, 1.0])
    shell = fem.ShellFemEnergy(
        fem.PointwiseShellAdapter[fem.KoiterStVK](),
        mapping,
        rest,
        [0.06] * len(rest),
        [theta] * len(rest),
        Fp=[Fp] * len(rest),
    )
    # All vertex DOFs are free, including rigid translations and rotations.
    with tbb_thread_limit(4):
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
    u = result.x
    Y = X + u.reshape(-1, 3)
    # Uniform Fp admits an affine, zero-energy target. Compare modulo rigid motion.
    target = X @ Fp.T
    U, _, Vt = np.linalg.svd((target - target.mean(0)).T @ (Y - Y.mean(0)))
    R = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
    target = (target - target.mean(0)) @ R + Y.mean(0)
    error = float(np.max(np.linalg.norm(Y - target, axis=1)))
    ev = shell.evaluate(u)
    residual = float(np.max(abs(ev.gradient())))
    assert error < 1e-6, error
    assert residual < 1e-5, residual
    for name, vertices in [("original", X), ("relaxed", Y), ("affine_target", target)]:
        mesh.write_obj(out / f"{name}.obj", mesh.TriangleMesh(vertices, faces))
    np.savez(out / "relaxed.npz", X=X, Y=Y, faces=faces)
    summary = dict(
        iterations=result.iterations,
        energy=ev.value,
        full_residual=residual,
        max_affine_target_error=error,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == "__main__":
    main()
