"""Recover shared positive diagonal shell Fp from equilibrium shapes and reactions.

Material and geometry are fixed; synthetic observations contain no noise.
"""

import numpy as np
from litepgo import dof, energy, fem, mesh, solver


def main():
    nodes = np.array([[x, y, 0.] for y in (-1., 0., 1.) for x in (-1., 0., 1.)])
    faces = []
    for y in range(2):
        for x in range(2):
            a = 3 * y + x
            faces.extend([[a, a + 1, a + 4], [a, a + 4, a + 3]])
    topology = mesh.TriangleTopology(9, np.array(faces, dtype=np.int32))
    stencils = mesh.make_triangle_mid_edge_stencils(topology)
    mapping = dof.StencilDofMap(9, stencils.vertices, 3)
    rest = [fem.TriangleMidEdgeRestData(mapping.gather(e, nodes.ravel()), stencils.opposite[e])
            for e in range(mapping.num_elements)]
    thickness = [0.1] * mapping.num_elements
    fixed_ids = np.array([i for i in range(27) if i // 3 != 4])
    fixed = dof.DofReduction(27, fixed_ids.tolist())
    X, Y = nodes[:, 0], nodes[:, 1]
    prescribed = np.column_stack([.08 * X + .025 * X**2 + .04 * Y,
                                  .03 * Y + .02 * X * Y, .08 * X * Y + .03 * X**2]).ravel()
    Elastic = fem.ShellFemEnergy[fem.KoiterStVK]
    Force = fem.ShellFemForceFpResponse[fem.KoiterStVK]

    theta = [np.array([2., 3.])] * mapping.num_elements
    truth = np.array([1.06, .95])
    n = len(truth)
    free_ids = np.setdiff1d(np.arange(nodes.size), fixed_ids)
    if not len(free_ids) or not len(fixed_ids):
        raise RuntimeError("Require both fixed and free DOFs")

    def plastic(p):
        with np.errstate(over="ignore", under="ignore"):
            stretches = np.exp(p)
        if not np.isfinite(stretches).all() or (stretches <= 0).any():
            raise ValueError("Unrepresentable plastic stretches")
        return np.diag([*stretches, 1.])

    def make_energy(p):
        Fp = plastic(p)
        return Elastic(mapping, rest, thickness, theta, [Fp] * mapping.num_elements)

    def partial(p, u):
        Fp = plastic(p)
        full = Force(dof.ConstantDofMap(topology, 9), mapping, rest, u, thickness, theta).eval_jacobian(Fp.ravel(order="F"))
        # Column-major diagonal slots; chain rule through exp(p).
        return full[:, np.arange(n) * 4].multiply(np.diag(Fp)[:n][None, :]).tocsc()

    def experiment(p, load, need_jacobian):
        initial = load * prescribed
        equilibrium = energy.StaticEquilibriumResponse(
            n, fixed, initial, solver.NewtonOptions(gradient_tolerance=1e-13, max_iterations=100),
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
        for h in (3e-5, 1e-5):
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
