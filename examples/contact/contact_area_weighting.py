"""Compare area weighting on fixed parallel sheets under uniform mesh refinement.

No equilibrium solve or coefficient conversion is performed. Run after building:
    uv run --no-sync python examples/contact/contact_area_weighting.py
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from litepgo import contact, mesh

REPO = Path(__file__).resolve().parents[2]
SIDE, GAP, DHAT, KAPPA = .16, .001, .002, 1e5


def parallel_sheets(n):
    vertices, triangles = [], []
    for z in (0., GAP):
        base = len(vertices)
        vertices.extend([[SIDE * i / n, SIDE * j / n, z]
                         for j in range(n + 1) for i in range(n + 1)])
        for j in range(n):
            for i in range(n):
                a = base + j * (n + 1) + i
                triangles.extend([[a, a + 1, a + n + 2], [a, a + n + 2, a + n + 1]])
    return mesh.TriangleMesh(np.array(vertices), np.array(triangles))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path,
                        default=REPO / "output/python/contact_area_weighting.csv")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["subdivisions", "edge_length", "triangles", "use_area_weighting", "energy", "normal_force"])
        for n in (4, 8, 16):
            surface = parallel_sheets(n)
            x = surface.positions.ravel()
            for weighted in (False, True):
                options = contact.IPCContactOptions(dhat=DHAT, use_area_weighting=weighted)
                barrier = contact.IPCContactEnergy(surface, options, contact.SpatialHash())
                value = KAPPA * barrier.eval_value(x)
                gradient = KAPPA * barrier.eval_gradient(x).reshape(-1, 3)
                force = abs(gradient[:(n + 1)**2, 2].sum())
                if (not np.isfinite([value, force]).all() or min(value, force) <= 0
                        or abs(gradient[:, 2].sum()) > 1e-10 * force):
                    raise RuntimeError("expected finite positive energy and balanced normal forces")
                writer.writerow([n, SIDE / n, len(surface.cells), int(weighted), value, force])
                print(f"n={n}, area_weighting={int(weighted)}, energy={value:.6g}, normal_force={force:.6g}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
