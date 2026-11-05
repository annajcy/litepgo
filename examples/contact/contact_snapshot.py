"""Export every active contact record on one mesh, without updating its Hessian pattern."""

import argparse
import csv
from pathlib import Path

import numpy as np

from litepgo import contact, mesh

REPO = Path(__file__).resolve().parents[2]


def contact_scene():
    # Five separated triangle pairs in one mesh: VV, VE, VF, near-parallel and parallel EE.
    positions = np.array([
        [[0, 0, 0], [2, 0, 0], [0, 2, 0], [-.2, -.3, .4], [-1, -2, 1], [-2, -1, 1]],
        [[0, 0, 0], [2, 0, 0], [0, 2, 0], [.7, -.2, .4], [2, -2, 1], [-2, -1, 1]],
        [[0, 0, 0], [2, 0, 0], [0, 2, 0], [.4, .3, .4], [2, 2, 1], [-1, 2, 1]],
        [[-1, 0, 0], [1, 0, 0], [0, -4, -2], [-.7, -.004, .2], [.8, .008, .2], [0, 4, 2]],
        [[-1, 0, 0], [1, 0, 0], [0, -4, -2], [-.7, 0, .2], [.8, 0, .2], [0, 4, 2]],
    ], dtype=float)
    positions[:, :, 0] += 10 * np.arange(5)[:, None]
    return mesh.TriangleMesh(positions.reshape(-1, 3), np.arange(30).reshape(-1, 3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=REPO / "output/python/contact_snapshot")
    args = parser.parse_args()
    surface = contact_scene()
    barrier = contact.IPCContactEnergy(surface, contact.IPCContactOptions(dhat=.7, min_distance=.05),
                                       contact.SpatialHash())
    positions = surface.positions.copy().reshape(5, 6, 3)
    positions[:, 3:, 2] -= .05
    x = positions.ravel()
    snapshot = barrier.contact_snapshot(x)
    count = len(snapshot.types)
    if (count != barrier.num_active_pairs(x) or barrier.hessian_pattern.nnz != 0
            or snapshot.distances.min() != barrier.min_active_distance(x)
            or set(snapshot.types) != {"VV", "VE", "VF", "EE"}
            or not np.any(snapshot.mollifiers == 0)):
        raise RuntimeError("expected the complete active set, all four kinds, and a retained zero-mollifier EE")
    args.output.mkdir(parents=True, exist_ok=True)
    mesh.write_obj(args.output / "mesh.obj", mesh.TriangleMesh(snapshot.positions, surface.cells))
    with (args.output / "contacts.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["type", "v0", "v1", "v2", "v3", "distance", "weight", "mollifier"])
        writer.writerows(zip(snapshot.types, *snapshot.vertex_ids.T, snapshot.distances,
                             snapshot.weights, snapshot.mollifiers))
    counts = ", ".join(f"{kind}={snapshot.types.count(kind)}" for kind in ("VV", "VE", "VF", "EE"))
    print(f"records={count}, {counts}, min_distance={snapshot.distances.min():.6g}, "
          f"zero_mollifiers={np.count_nonzero(snapshot.mollifiers == 0)}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
