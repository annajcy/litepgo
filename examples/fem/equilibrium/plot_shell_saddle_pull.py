"""Compare a flat shell and its saddle equilibrium from matching OBJ files."""

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
    parser.add_argument("rest", type=Path, help="initial shell OBJ")
    parser.add_argument("equilibrium", type=Path, help="deformed shell OBJ")
    parser.add_argument(
        "-o", "--output", type=Path, help="PNG path, defaulting to the equilibrium stem"
    )
    args = parser.parse_args()

    rest, equilibrium = (mesh.read_obj(path) for path in (args.rest, args.equilibrium))
    original, result = rest.positions, equilibrium.positions
    faces = rest.cells
    if (
        not len(faces)
        or original.shape != result.shape
        or not np.array_equal(faces, equilibrium.cells)
    ):
        parser.error(
            "both OBJs must share the same nonempty topology and vertex ordering"
        )

    displacement = (result[:, 2] - original[:, 2])[faces].mean(axis=1)
    magnitude = max(np.abs(displacement).max(), np.finfo(float).eps)
    scale = matplotlib.colors.Normalize(vmin=-magnitude, vmax=magnitude)
    cmap = plt.get_cmap("coolwarm")
    combined = np.vstack((original, result))
    lo, hi = combined.min(axis=0), combined.max(axis=0)
    padding = 0.05 * max(hi - lo)
    lo, hi = lo - padding, hi + padding

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(8, 4),
        subplot_kw={"projection": "3d", "proj_type": "ortho"},
        layout="constrained",
    )
    figure.get_layout_engine().set(w_pad=0.22)

    for axis, title, points, values in zip(
        axes,
        ("Initial sheet", "Saddle equilibrium"),
        (original, result),
        (np.zeros_like(displacement), displacement),
    ):
        axis.add_collection3d(
            Poly3DCollection(
                points[faces],
                facecolors=cmap(scale(values)),
                edgecolors="#465564",
                linewidths=0.45,
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
        axis.view_init(elev=27, azim=-60)
    figure.colorbar(
        matplotlib.cm.ScalarMappable(norm=scale, cmap=cmap),
        ax=axes,
        orientation="horizontal",
        shrink=0.55,
        label="Mean z displacement",
    )

    output = args.output or args.equilibrium.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
