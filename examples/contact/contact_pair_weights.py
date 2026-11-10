"""Three arbitrary primitive groups: compare pair multipliers 0.5, 2 and 0."""

import argparse
import csv
from pathlib import Path

import numpy as np
from litepgo import contact, mesh

REPO = Path(__file__).resolve().parents[2]
GROUPS = ((0, 1), (1, 2), (0, 2))
TABLE = np.array([[1., .5, 0], [.5, 1, 2], [0, 2, 1]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=REPO / "output/python/contact_pair_weights.csv")
    args = parser.parse_args()
    triangle_pair = np.array([[0., 0, 0], [10, 0, 0], [0, 10, 0],
                             [2, 2, .5], [6, 2, 4.5], [2, 6, 4.5]])
    positions = np.concatenate([triangle_pair + [30*i, 0, 0] for i in range(3)])
    surface = mesh.TriangleMesh(positions, np.arange(18).reshape(-1, 3))
    vertex_groups = np.repeat(np.array(GROUPS).ravel(), 3)
    edges = contact.contact_edges(surface.topology)
    assert np.all(vertex_groups[edges[:, 0]] == vertex_groups[edges[:, 1]])
    pair_weights = contact.ContactPairWeights(vertex_groups, vertex_groups[edges[:, 0]],
                                               np.array(GROUPS).ravel(), TABLE)
    options = contact.IPCContactOptions(dhat=.7, min_distance=.05)
    reference = contact.IPCContactEnergy(surface, options, contact.SpatialHash())
    grouped = contact.IPCContactEnergy(surface, options, contact.SpatialHash(), pair_weights)
    x = positions.ravel()
    snapshots = [e.contact_snapshot(x) for e in (reference, grouped)]
    gradients = [e.evaluate(x).gradient().reshape(-1, 3) for e in (reference, grouped)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    totals = np.zeros(2)
    with args.output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["pair", "group_a", "group_b", "multiplier", "reference_energy", "weighted_energy",
                         "reference_force", "weighted_force", "active_records"])
        for pair, (a, b) in enumerate(GROUPS):
            values, counts = [], []
            for snapshot in snapshots:
                rows = snapshot.vertex_ids[:, 0] // 6 == pair
                d = snapshot.distances[rows]**2 - options.min_distance**2
                h = options.dhat * (options.dhat + 2 * options.min_distance)
                values.append(np.sum(snapshot.weights[rows] * snapshot.mollifiers[rows] * -(d-h)**2 * np.log(d/h)))
                counts.append(np.count_nonzero(rows))
            forces = [abs(g[6*pair+3:6*pair+6, 2].sum()) for g in gradients]
            scale = TABLE[a, b]
            if (counts != [1, int(scale > 0)] or not np.allclose([values[1], forces[1]], scale * np.array([values[0], forces[0]]), rtol=1e-12, atol=1e-14)):
                raise RuntimeError("expected each pair's energy and force to follow its multiplier")
            writer.writerow([pair, a, b, scale, *values, *forces, counts[1]])
            totals += values
            print(f"groups=({a},{b}), multiplier={scale:g}, energy={values[1]:.6g}, force={forces[1]:.6g}, records={counts[1]}")
    np.testing.assert_allclose(totals, [reference.evaluate(x).value, grouped.evaluate(x).value], rtol=1e-12)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
