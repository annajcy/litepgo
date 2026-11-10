"""Fit shared Neo-Hookean parameters from forces at a prescribed deformation, using batches.

Run from the repository root after building litepgo:
    uv run --no-sync python examples/energy/response/mesh_response.py
"""

from litepgo.energy import AffineResponse
import argparse
from statistics import median
from time import perf_counter

import numpy as np
from litepgo import dof, fem, mesh, energy, solver


def problem(copies=1):
    # Each copy has two tetrahedra sharing a face; copies are disconnected.
    X = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    cells = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32)
    F = np.array([[1.15, 0.12, 0.0], [0.03, 0.94, 0.08], [0.0, 0.02, 1.07]])
    cells = (cells[None] + 5 * np.arange(copies)[:, None, None]).reshape(-1, 4)
    topology = mesh.TetTopology(5 * copies, cells)
    nodes = np.tile(X, (copies, 1))
    mapping = dof.VertexDofMap(topology, 3)
    Field = fem.InterpolatedResponse[fem.TetP1Basis]
    geometry = [
        Field(mapping.gather(e, nodes.ravel())) for e in range(topology.num_cells)
    ]
    displacement = (nodes @ (F - np.eye(3)).T).ravel()
    plastic = [AffineResponse(np.zeros((9, 3)), np.eye(3).ravel())] * topology.num_cells
    return topology, mapping, geometry, displacement, plastic


def make_responses(topology, mapping, geometry, displacement, plastic, batch_size):
    parameters = dof.ConstantDofMap(topology, 2)
    Force = fem.VolumetricFemForceMaterialResponse[fem.TetP1Basis, fem.TetCentroidQuadrature]
    return (
        Force(fem.PointwiseVolumetricAdapter[fem.NeoHookean](), parameters, mapping, geometry, displacement, plastic, batch_size=1),
        Force(fem.PointwiseVolumetricAdapter[fem.NeoHookean](),
            parameters, mapping, geometry, displacement, plastic, batch_size=batch_size
        ),
    )


class ForceResidual(energy.FixedPatternResponse):
    def __init__(self, force, observed):
        super().__init__(np.ones((force.num_outputs, force.num_inputs)))
        self.force, self.observed = force, observed

    def eval_impl(self, theta, need_jacobian):
        evaluation = self.force.evaluate(theta)
        J = evaluation.jacobian() if need_jacobian else None
        value = evaluation.value
        residual = value - self.observed
        return (residual, J) if need_jacobian else residual


def run(copies=1, batch_size=256, repeats=0):
    data = problem(copies)
    topology, mapping, geometry, displacement, plastic = data
    single, batched = make_responses(*data, batch_size)
    truth = np.array([2.0, 3.0])
    # Independent observations from the energy API, at the same fixed displacement.
    Energy = fem.VolumetricFemEnergy[fem.TetP1Basis, fem.TetCentroidQuadrature]
    energy = Energy(fem.PointwiseVolumetricAdapter[fem.NeoHookean](),
        mapping, geometry, plastic, [AffineResponse(np.zeros((2, 3)), truth)] * topology.num_cells
    )
    expected = energy.evaluate(displacement).gradient()
    single_evaluation = single.evaluate(truth)
    J = single_evaluation.jacobian()
    single_value = single_evaluation.value
    np.testing.assert_allclose(single_value, expected, atol=1e-13)
    batched_evaluation = batched.evaluate(truth)
    batch_J = batched_evaluation.jacobian()
    value = batched_evaluation.value
    np.testing.assert_allclose(value, expected, atol=1e-14)
    np.testing.assert_array_equal(batch_J.indptr, J.indptr)
    np.testing.assert_array_equal(batch_J.indices, J.indices)
    np.testing.assert_allclose(batch_J.data, J.data, atol=1e-14)
    assert np.linalg.matrix_rank(J[:15].toarray()) == 2
    residual = ForceResidual(batched, expected)
    options = solver.GaussNewtonOptions(gradient_tolerance=1e-10)
    result = solver.gauss_newton_solve(residual, np.array([0.7, 1.2]), options)
    assert result.status == solver.GaussNewtonStatus.Converged
    np.testing.assert_allclose(result.x, truth, atol=1e-7, rtol=0)
    print(f"material fit: lambda={result.x[0]:.6f}, mu={result.x[1]:.6f}")
    if repeats:
        print(f"elements={2 * copies}, batch_size={batch_size}, repeats={repeats}")
        for need in (False, True):
            times = []
            for obj in (single, batched):
                for _ in range(3):
                    evaluation = obj.evaluate(truth)
                    if need:
                        evaluation.jacobian()
                    evaluation.value
                samples = []
                for _ in range(repeats):
                    start = perf_counter()
                    evaluation = obj.evaluate(truth)
                    if need:
                        evaluation.jacobian()
                    evaluation.value
                    samples.append(perf_counter() - start)
                times.append(1000 * median(samples))
            print(
                f"{'value+J' if need else 'value'}: batch1={times[0]:.3f} ms, "
                f"batchN={times[1]:.3f} ms, speedup={times[0] / times[1]:.2f}x"
            )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--copies", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--repeats",
        type=int,
        default=0,
        help="benchmark warm evaluation, median milliseconds",
    )
    args = parser.parse_args()
    if args.copies < 1 or args.batch_size < 1 or args.repeats < 0:
        parser.error("copies/batch-size must be positive; repeats must be nonnegative")
    run(args.copies, args.batch_size, args.repeats)
