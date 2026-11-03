"""Compare the rest and equilibrium OBJ meshes from differential shell growth."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

from litepgo import mesh

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="directory containing rest.obj and equilibrium.obj")
    parser.add_argument("-o", "--output", type=Path, help="PNG path, defaulting to shell_plastic_field.png")
    args = parser.parse_args()

    rest = mesh.read_obj(args.input / "rest.obj")
    equilibrium = mesh.read_obj(args.input / "equilibrium.obj")
    original, result = rest.positions, equilibrium.positions
    faces = rest.cells
    if not len(faces) or original.shape != result.shape or not np.array_equal(faces, equilibrium.cells):
        parser.error("both OBJs must share the same nonempty topology and vertex ordering")

    combined = np.vstack((original, result))[:, :2]
    lo, hi = combined.min(axis=0), combined.max(axis=0)
    padding = 0.05 * max(hi - lo)
    lo, hi = lo - padding, hi + padding
    # The first two x columns are clamped in the solver.
    clamped = original[:, 0] <= np.unique(original[:, 0])[1]

    figure, axes = plt.subplots(2, 1, figsize=(8, 5.5), layout="constrained")
    for axis, title, points in zip(axes, ("Initial strip", "Growth equilibrium"), (original, result)):
        axis.add_collection(PolyCollection(
            points[faces, :2], facecolors="#c2dce6", edgecolors="#456476", linewidths=0.6
        ))
        axis.scatter(*points[clamped, :2].T, color="#176b87", s=16, label="Clamped vertices")
        axis.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), xlabel="x", ylabel="y", title=title)
        axis.set_aspect("equal", adjustable="box")
    axes[0].legend(loc="upper right")

    output = args.output or args.input / "shell_plastic_field.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
