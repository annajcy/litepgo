"""Run worker/GIL checks in a subprocess so a deadlock fails with a timeout."""
import subprocess
import sys
import textwrap
from pathlib import Path


def test_batch_workers_gil_exception_and_recovery():
    code = """
        import threading
        import time
        import numpy as np
        from litepgo import dof, energy
        from batch_fixtures import BatchEnergyFixture, BatchResponseFixture

        mapping = dof.StencilDofMap(1, [[0]] * 64, 1)
        workers = set()
        saved = []
        fail = False

        def visit(ids, x):
            workers.add(threading.get_ident())
            saved.append((x, x.copy()))
            time.sleep(.001)  # Release the GIL while another batch enters Python.
            if fail and 20 in ids:
                raise KeyError("worker failure")

        def e(ids, x, need_g, need_h):
            visit(ids, x)
            return .5 * (x*x).sum(axis=1), x if need_g else None, np.ones_like(x) if need_h else None

        def step(ids, x, dx):
            visit(ids, x)
            return np.full(len(ids), .5)

        def r(ids, x, need_j):
            visit(ids, x)
            return (x, np.ones_like(x)) if need_j else x

        E = BatchEnergyFixture(mapping, e, batch_size=1, step=step)
        R = BatchResponseFixture(mapping, mapping, r, batch_size=1)
        x = np.ones(1)
        for operation, expected in [
            (lambda: E.evaluate(x).value, 32),
            (lambda: E.evaluate(x).gradient()[0], 64),
            (lambda: E.evaluate(x).hessian()[0, 0], 64),
            (lambda: E.evaluate(x).max_feasible_step(x), .5),
            (lambda: R.evaluate(x).value[0], 64),
            (lambda: R.evaluate(x).jacobian()[0, 0], 64),
        ]:
            workers.clear()
            assert operation() == expected
            assert len(workers) > 1
            fail = True
            try:
                operation()
            except KeyError as error:
                assert "worker failure" in str(error)
            else:
                raise AssertionError("missing worker exception")
            fail = False
            assert operation() == expected
        for actual, expected in saved:
            np.testing.assert_array_equal(actual, expected)
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True, text=True, timeout=30, cwd=Path(__file__).parent,
    )
    assert result.returncode == 0, result.stdout + result.stderr
