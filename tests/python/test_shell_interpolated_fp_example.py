"""Fast interpolation checks; the executable example checks the full solve."""

import importlib.util
from pathlib import Path


def test_plastic_gradient_interpolation():
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/fem/material/shell_interpolated_fp.py"
    )
    spec = importlib.util.spec_from_file_location("shell_interpolated_fp", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    example.check_fields()


def test_analytic_twist_gradient():
    import numpy as np

    path = (
        Path(__file__).resolve().parents[2]
        / "examples/fem/material/shell_interpolated_fp.py"
    )
    spec = importlib.util.spec_from_file_location("shell_interpolated_fp", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    lo = np.array([-1.0, 2.0, 0.3])
    hi = lo + [2.0, 3.0, 4.0]
    points = lo + np.random.default_rng(71).uniform(0.1, 0.9, (12, 3)) * (hi - lo)

    def position(X):
        r = X - (lo + hi) / 2
        angle = np.pi * r[:, 2] / (3 * (hi[2] - lo[2]))
        c, s = np.cos(angle), np.sin(angle)
        return np.column_stack(
            [c * r[:, 0] - s * r[:, 1], s * r[:, 0] + c * r[:, 1], r[:, 2]]
        )

    h = 1e-6
    fd = np.stack(
        [
            (position(points + h * v) - position(points - h * v)) / (2 * h)
            for v in np.eye(3)
        ],
        axis=2,
    )
    exact = example.analytic_fp(points, lo, hi)
    np.testing.assert_allclose(exact, fd, atol=2e-9, rtol=2e-9)
    np.testing.assert_allclose(np.linalg.det(exact), 1.0, atol=1e-14)
