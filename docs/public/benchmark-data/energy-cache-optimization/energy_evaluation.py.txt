"""Scalar and FEM EnergyEvaluation timings with deterministic numerical checks.

Run after correctness tests, with no concurrent builds/tests. PYTHONPATH selects
an installed build; the archived pre-migration script uses the former API.
"""
import argparse
import json
import statistics
import time

import numpy as np
import scipy.sparse as sp
from litepgo import energy, tbb_thread_limit
from equilibrium_evaluation import problem


def evaluate(model, x, operation):
    state = model.evaluate(x)
    if operation == 'value':
        return (state.value,)
    if operation == 'gradient':
        gradient = state.gradient()
        return state.value, gradient
    hessian = state.projected_hessian() if operation.startswith('projected') else state.hessian()
    if operation.endswith('_only'):
        return None, None, hessian
    return state.value, state.gradient(), hessian


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=15)
    args = parser.parse_args()
    for threads in (1, 4):
        with tbb_thread_limit(threads):
            cases = []
            if threads == 1:
                for n in (3, 1000):
                    cases.append((f'quadratic_{n}', energy.QuadraticEnergy(sp.eye(n, format='csc')), np.ones(n), 1000))
            for family in ('tet', 'hermite', 'shell'):
                eq, p, potential, _ = problem(family, 2, False, {'solves': 0})
                x = eq.evaluate(p).value
                cases.append((family, potential(p), x, 20))
            for name, model, x, repeats in cases:
                for operation in ('value', 'gradient', 'exact', 'projected', 'hessian_only', 'projected_only'):
                    reference = evaluate(model, x, operation)
                    samples = []
                    for _ in range(args.samples):
                        start = time.perf_counter()
                        for _ in range(repeats):
                            actual = evaluate(model, x, operation)
                        samples.append((time.perf_counter() - start) * 1e6 / repeats)
                        for a, b in zip(actual, reference):
                            if a is None:
                                assert b is None
                            elif sp.issparse(a):
                                np.testing.assert_allclose(a.toarray(), b.toarray(), atol=1e-12, rtol=1e-10)
                            else:
                                np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-10)
                    print(json.dumps(dict(case=name, threads=threads, operation=operation,
                        median_us=statistics.median(samples), samples_us=samples,
                        value=reference[0], gradient=reference[1].tolist() if len(reference)>1 and reference[1] is not None else None,
                        hessian_action=(reference[2] @ np.column_stack((np.sin(np.arange(len(x)) + .3), np.cos(np.arange(len(x)) + .7)))).tolist() if len(reference)>2 else None)), flush=True)


if __name__ == '__main__':
    main()
