"""Compare the rest, free-top, and clamped-top OBJ outputs of Tet P1 plastic twist."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

from litepgo import mesh

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stem", type=Path, help="output path stem shared by the three OBJ files"
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="PNG path, defaulting to <stem>.png"
    )
    args = parser.parse_args()

    cases = (("rest", "Initial"), ("free", "Free top"), ("clamped", "Clamped top"))
    surfaces = [
        mesh.read_obj(args.stem.with_name(f"{args.stem.name}_{case}.obj"))
        for case, _ in cases
    ]
    faces = surfaces[0].cells
    positions = [surface.positions for surface in surfaces]
    if not len(faces) or any(
        points.shape != positions[0].shape or not np.array_equal(surface.cells, faces)
        for surface, points in zip(surfaces[1:], positions[1:])
    ):
        parser.error(
            "the three OBJs must share the same nonempty boundary topology and vertex ordering"
        )

    z = positions[0][faces, 2]
    tolerance = 1e-8 * np.ptp(z)
    colors = np.full(len(faces), "#c2dce6")
    colors[(np.abs(z - z.min()) <= tolerance).all(axis=1)] = "#176b87"
    colors[(np.abs(z - z.max()) <= tolerance).all(axis=1)] = "#d47a24"

    combined = np.concatenate(positions)
    lo, hi = combined.min(axis=0), combined.max(axis=0)
    padding = 0.05 * max(hi - lo)
    lo, hi = lo - padding, hi + padding

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(10, 3.8),
        subplot_kw={"projection": "3d", "proj_type": "ortho"},
        layout="constrained",
    )
    for axis, (_, title), points in zip(axes, cases, positions):
        axis.add_collection3d(
            Poly3DCollection(
                points[faces], facecolors=colors, edgecolors="#456476", linewidths=0.35
            )
        )
        axis.set(
            xlim=(lo[0], hi[0]),
            ylim=(lo[1], hi[1]),
            zlim=(lo[2], hi[2]),
            xlabel="x",
            ylabel="y",
            zlabel="z",
            title=title,
        )
        axis.set_box_aspect(hi - lo)
        axis.view_init(elev=22, azim=-60)

    output = args.output or args.stem.with_name(f"{args.stem.name}.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
