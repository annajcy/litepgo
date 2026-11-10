"""Open, traction-free Schwarz P cell: no PBC, no contact, only a rigid gauge."""

import json
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from litepgo import fem, dof, mesh, solver, energy, tbb_thread_limit
from litepgo.linalg import sparse_solver


def rigid_frame_map(X):
    """Remove six rigid freedoms without prescribing edge lengths."""
    n = len(X)
    # Fix a rigid frame without fixing any triangle lengths: A fixed, B along AB,
    # C in ABC plane. All remaining vertices have three unrestricted coordinates.
    a = int(np.argmin(X[:, 0] + X[:, 1] + X[:, 2]))
    b = int(np.argmax(np.linalg.norm(X - X[a], axis=1)))
    e1 = (X[b] - X[a]) / np.linalg.norm(X[b] - X[a])
    c = int(np.argmax(np.linalg.norm(np.cross(X - X[a], e1), axis=1)))
    e3 = np.cross(e1, X[c] - X[a])
    e3 /= np.linalg.norm(e3)
    e2 = np.cross(e3, e1)
    rows = []
    cols = []
    values = []
    col = 0
    for i in range(n):
        directions = (
            [] if i == a else [e1] if i == b else [e1, e2] if i == c else np.eye(3)
        )
        for direction in directions:
            for k in range(3):
                rows.append(3 * i + k)
                cols.append(col)
                values.append(direction[k])
            col += 1
    T = coo_matrix((values, (rows, cols)), shape=(3 * n, col)).tocsr()
    assert col == 3 * n - 6
    return T, (a, b, c)


def main():
    root = Path(__file__).resolve().parents[3]
    p = root / "output/schwarz_p_fp_free"
    p.mkdir(parents=True, exist_ok=True)
    cell_mesh = mesh.read_obj(root / "assets/mesh/schwarz_p_open_cell.obj")
    X = np.array(cell_mesh.positions)
    faces = np.array(cell_mesh.cells)
    n = len(X)
    bounds = np.array([X.min(axis=0), X.max(axis=0)])
    stencils = mesh.make_triangle_mid_edge_stencils(cell_mesh.topology)
    mapping = dof.StencilDofMap(n, stencils.vertices, 3)
    rest = [
        fem.TriangleMidEdgeRestData(mapping.gather(i, X.ravel()), stencils.opposite[i])
        for i in range(len(faces))
    ]
    T, (a, b, c) = rigid_frame_map(X)
    col = T.shape[1]
    theta = np.array([1e8 * 0.35 / (1 - 0.35**2), 1e8 / (2 * 1.35)])
    center = X[faces].mean(axis=1)
    q = 0.65 * np.cos(
        2 * np.pi * (center[:, 2] - bounds[0, 2]) / (bounds[1, 2] - bounds[0, 2])
    )
    material = fem.PointwiseShellAdapter[fem.KoiterStVK]()
    options = solver.NewtonOptions(
        max_iterations=1500, project_hessian=False, max_coordinate_step=0.01
    )
    mesh.write_obj(p / "original.obj", cell_mesh)
    z = np.zeros(col)
    records = []
    with tbb_thread_limit(4):
        Fp = np.tile(np.eye(3), (len(faces), 1, 1))
        Fp[:, 0, 0] = np.exp(q)
        Fp[:, 1, 1] = np.exp(-q)
        E = fem.ShellFemEnergy(
            material, mapping, rest, [0.06] * len(faces), [theta] * len(faces), Fp=Fp
        )
        initial = E.evaluate(T @ z).value
        reduced = energy.AffineMappedEnergy(E, T, np.zeros(3 * n))

        def progress(step):
            if step.iteration % 20 == 0:
                print(
                    f"iteration {step.iteration}: E={step.energy}, residual={step.gradient_norm}",
                    flush=True,
                )

        solved = solver.newton_solve(
            reduced,
            dof.DofReduction(T.shape[1], []),
            z,
            options,
            convergence=solver.AbsoluteConvergence(1e-5),
            on_iteration=progress,
        )
        result = SimpleNamespace(
            solver_result=solved,
            converged=solved.status == solver.NewtonStatus.Converged,
            fluctuation=solved.x,
            displacement=(T @ solved.x).reshape(-1, 3),
            energy=solved.energy,
        )
        print(
            "status",
            result.solver_result.status,
            "iterations",
            result.solver_result.iterations,
            flush=True,
        )
        assert result.converged, result.solver_result.status
        z = result.fluctuation
        Y = X + result.displacement
        ev = E.evaluate(result.displacement.ravel())
        g = ev.gradient()
        H = (T.T @ ev.hessian() @ T).tocsc()
        chol_type = getattr(sparse_solver, "AccelerateLLT", None) or getattr(
            sparse_solver, "PardisoLLT", None
        )
        spd = bool(chol_type().compute(H)) if chol_type else None
        entry = dict(
            initial_energy=initial,
            energy=result.energy,
            iterations=result.solver_result.iterations,
            reduced_residual=float(np.max(abs(T.T @ g))),
            full_residual=float(np.max(abs(g))),
            internal_hessian_spd=spd,
            max_displacement=float(np.max(np.linalg.norm(Y - X, axis=1))),
        )
        records.append(entry)
        print(json.dumps(entry), flush=True)
        np.savez(
            p / "relaxed.npz", X=X, Y=Y, faces=faces, z=z, Fp=Fp, q=q, anchors=[a, b, c]
        )
        (p / "summary.json").write_text(json.dumps(records, indent=2))
        mesh.write_obj(p / "relaxed.obj", mesh.TriangleMesh(Y, faces))
    assert records[-1]["full_residual"] < 1e-5
    assert records[-1]["energy"] < records[-1]["initial_energy"]
    print("COMPLETE", p, flush=True)


if __name__ == "__main__":
    main()
