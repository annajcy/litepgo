"""Focused evaluation-cache and small-assembly timings; PYTHONPATH selects the build."""
import argparse
import json
import statistics
import time

import numpy as np
import scipy.sparse as sp
from litepgo import energy, tbb_thread_limit
from equilibrium_evaluation import problem


def measure(name, model, x, operation, threads, samples, repeats):
    def run():
        state = model.evaluate(x)
        if operation == 'create':
            return state
        if operation == 'gradient':
            return state.gradient()
        return state.hessian()

    for _ in range(50):
        run()
    reference = run()
    times = []
    for _ in range(samples):
        start = time.perf_counter_ns()
        for _ in range(repeats):
            actual = run()
        times.append((time.perf_counter_ns() - start) / (1000 * repeats))
        if operation == 'gradient':
            np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-12)
        elif operation == 'hessian':
            assert (actual - reference).nnz == 0
    # Numerical fingerprints are outside the timed region and checked across builds.
    e = model.evaluate(x)
    H = e.hessian()
    probe = np.sin(np.arange(len(x)) + .3)
    print(json.dumps(dict(case=name, threads=threads, operation=operation,
        median_us=statistics.median(times), samples_us=times, value=e.value,
        gradient=e.gradient().tolist(), hessian_action=(H @ probe).tolist())), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['composites', 'assembly'], required=True)
    parser.add_argument('--samples', type=int, default=21)
    args = parser.parse_args()
    if args.suite == 'composites':
        with tbb_thread_limit(1):
            for n in (3, 1000, 10000):
                A = sp.diags([-.1*np.ones(n-1), np.ones(n), -.1*np.ones(n-1)], [-1, 0, 1], format='csc')
                leaf = energy.QuadraticEnergy(A)
                pair = energy.WeightedSumEnergy([(leaf, .75), (leaf, .25)])
                nested = energy.WeightedSumEnergy([(pair, .5), (pair, .5)])
                mapped = energy.AffineMappedEnergy(pair, sp.eye(n, format='csc') * 1.1)
                x = np.sin(np.arange(n) + .1)
                for name, model in [('leaf', leaf), ('pair', pair), ('nested', nested), ('mapped', mapped)]:
                    for operation in ('create', 'gradient', 'hessian'):
                        measure(f'{name}/{n}', model, x, operation, 1, args.samples,
                                100 if operation != 'hessian' else 20)
    else:
        for threads in (1, 4):
            with tbb_thread_limit(threads):
                for family, sizes in [('shell', (1, 2, 4, 8)), ('tet', (1, 2)), ('hermite', (1, 2))]:
                    for n in sizes:
                        eq, p, potential, _ = problem(family, n, False, {'solves': 0})
                        x = eq.evaluate(p).value
                        model = potential(p)
                        for operation in ('gradient', 'hessian'):
                            measure(f'{family}/{n}', model, x, operation, threads, args.samples,
                                    100 if operation == 'gradient' else 20)


if __name__ == '__main__':
    main()
