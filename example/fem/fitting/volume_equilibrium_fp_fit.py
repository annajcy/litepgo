"""Recover shared positive diagonal volume Fp from equilibrium shapes and reactions.

Material and geometry are fixed; synthetic observations contain no noise.
"""
from pathlib import Path

import numpy as np
from litepgo import dof, energy, fem, mesh, solver


def main():
    volume = mesh.read_veg(Path(__file__).resolve().parents[3] / "assets/box-tet.veg")
    topology, nodes = volume.topology, volume.positions
    mapping = dof.VertexDofMap(topology, 3)
    XField = fem.InterpolatedField3D[fem.TetP1Basis]
    geometry = [XField(mapping.gather(e, nodes.ravel())) for e in range(mapping.num_elements)]
    z = nodes[:, 2]
    height = z.max() - z.min()
    if not height > 0:
        raise RuntimeError("Require nonempty volume with positive height")
    top = np.flatnonzero(np.abs(z - z.max()) < height * 1e-8)
    boundary = np.flatnonzero((np.abs(z - z.min()) < height * 1e-8) | (np.abs(z - z.max()) < height * 1e-8))
    fixed_ids = (3 * boundary[:, None] + np.arange(3)).ravel()
    fixed = dof.DofReduction(nodes.size, fixed_ids.tolist())
    prescribed = np.zeros(nodes.size)
    prescribed[3 * top] = .04 * height
    prescribed[3 * top + 2] = .05 * height
    Elastic = fem.VolumetricFemEnergy[fem.NeoHookean, fem.TetP1Basis, fem.TetCentroidQuadrature]
    Force = fem.VolumetricFemForceFpResponse[fem.NeoHookean, fem.TetP1Basis, fem.TetCentroidQuadrature]
    theta = [fem.IdentityField3D(np.array([2., 3.])) for _ in geometry]
    truth = np.array([1.04, .97, 1.02])
    n = len(truth)
    free_ids = np.setdiff1d(np.arange(nodes.size), fixed_ids)
    if not len(free_ids) or not len(fixed_ids):
        raise RuntimeError("Require both fixed and free DOFs")

    def plastic(p):
        with np.errstate(over="ignore", under="ignore"):
            stretches = np.exp(p)
        if not np.isfinite(stretches).all() or (stretches <= 0).any():
            raise ValueError("Unrepresentable plastic stretches")
        return np.diag(stretches)

    def make_energy(p):
        Fp = plastic(p)
        return Elastic(mapping, geometry, [fem.IdentityField3D(Fp.ravel(order="F")) for _ in geometry], theta)

    def partial(p, u):
        Fp = plastic(p)
        full = Force(dof.ConstantDofMap(topology, 9), mapping, geometry, u, theta).eval_jacobian(Fp.ravel(order="F"))
        # Column-major diagonal slots; chain rule through exp(p).
        return full[:, np.arange(n) * 4].multiply(np.diag(Fp)[:n][None, :]).tocsc()

    def experiment(p, load, need_jacobian):
        initial = load * prescribed
        equilibrium = energy.StaticEquilibriumResponse(
            n, fixed, initial, solver.NewtonOptions(gradient_tolerance=1e-11, max_iterations=100),
            make_energy, partial,
        )
        u, J = equilibrium.eval(p, need_jacobian=need_jacobian)
        elastic = make_energy(p)
        gradient = elastic.eval_gradient(u)
        if not np.array_equal(u[fixed_ids], initial[fixed_ids]) or np.max(np.abs(gradient[free_ids])) > 2e-11:
            raise RuntimeError("Invalid equilibrium or changed boundary values")
        value = np.concatenate([u[free_ids], gradient[fixed_ids]])
        jacobian = None
        if need_jacobian:
            total = partial(p, u) + elastic.eval_hessian(u) @ J
            jacobian = np.vstack([J.toarray()[free_ids], total.toarray()[fixed_ids]])
            if np.linalg.norm(jacobian[:len(free_ids)]) < 1e-5:
                raise RuntimeError("Experiment must have nonzero equilibrium sensitivity")
        return value, jacobian

    target, _ = experiment(np.log(truth), 1., False)
    scales = np.concatenate([np.full(len(free_ids), max(np.linalg.norm(target[:len(free_ids)]), 1e-8)),
                             np.full(len(fixed_ids), max(np.linalg.norm(target[len(free_ids):]), 1e-8))])

    class Residual(energy.FixedPatternResponse):
        def __init__(self):
            super().__init__(np.ones((len(target), n)))

        def eval_values(self, p, need_jacobian):
            value, J = experiment(p, 1., need_jacobian)
            r = (value - target) / scales
            return (r, J / scales[:, None]) if need_jacobian else r

    residual = Residual()

    def check_rank(p):
        singular = np.linalg.svd(residual.eval_jacobian(p).toarray(), compute_uv=False)
        ratio = singular[-1] / singular[0]
        if not ratio > 1e-6:
            raise RuntimeError("Observations cannot identify every plastic stretch")
        return ratio

    initial = np.zeros(n)
    initial_ratio = check_rank(initial)
    truth_ratio = check_rank(np.log(truth))
    state_fd_error = reaction_fd_error = 0.
    for p in (initial, np.log(truth)):
        _, J = experiment(p, 1., True)
        for h in (1e-4, 1e-5):
            for c in range(n):
                delta = np.eye(n)[c] * h
                forward, _ = experiment(p + delta, 1., False)
                backward, _ = experiment(p - delta, 1., False)
                error = ((forward - backward) / (2 * h) - J[:, c]) / scales
                state_fd_error = max(state_fd_error, np.max(np.abs(error[:len(free_ids)])))
                reaction_fd_error = max(reaction_fd_error, np.max(np.abs(error[len(free_ids):])))
    if max(state_fd_error, reaction_fd_error) > 2e-6:
        raise RuntimeError(f"Re-solve FD failed: state={state_fd_error}, reaction={reaction_fd_error}")
    result = solver.gauss_newton_solve(residual, initial, solver.GaussNewtonOptions(gradient_tolerance=1e-10))
    fitted = np.exp(result.x)
    fitted_ratio = check_rank(result.x)
    parameter_error = np.linalg.norm(fitted - truth) / np.linalg.norm(truth)
    relative_residual = np.linalg.norm(residual.eval_value(result.x))
    held_out, _ = experiment(np.log(truth), .6, False)
    prediction, _ = experiment(result.x, .6, False)
    held_out_error = np.linalg.norm((prediction - held_out) / scales)
    converged = result.status == solver.GaussNewtonStatus.Converged
    print("truth=" + " ".join(f"{x:.12g}" for x in truth) + ", fitted=" + " ".join(f"{x:.12g}" for x in fitted)
          + f", converged={int(converged)}, parameter_error={parameter_error:.12g}, relative_residual={relative_residual:.12g}"
          + f", held_out_error={held_out_error:.12g}, state_fd_error={state_fd_error:.12g}, reaction_fd_error={reaction_fd_error:.12g}"
          + f", initial_ratio={initial_ratio:.12g}, truth_ratio={truth_ratio:.12g}, fitted_ratio={fitted_ratio:.12g}")
    return 0 if converged and parameter_error < 1e-6 and relative_residual < 1e-8 and held_out_error < 1e-8 else 1


if __name__ == "__main__":
    raise SystemExit(main())
