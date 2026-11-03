"""Compare the rest, noisy, and smoothed OBJ meshes from Laplacian optimization."""

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
    parser.add_argument("input", type=Path, help="directory containing rest.obj, noisy.obj, and smoothed.obj")
    parser.add_argument("-o", "--output", type=Path, help="PNG path, defaulting to laplacian_optimization.png")
    args = parser.parse_args()

    cases = (("rest", "Original"), ("noisy", "Noisy"), ("smoothed", "Smoothed"))
    surfaces = [mesh.read_obj(args.input / f"{name}.obj") for name, _ in cases]
    positions = [surface.positions for surface in surfaces]
    faces = surfaces[0].cells
    if not len(faces) or any(
        points.shape != positions[0].shape or not np.array_equal(surface.cells, faces)
        for surface, points in zip(surfaces[1:], positions[1:])
    ):
        parser.error("the three OBJ files must share the same nonempty topology and vertex ordering")
    displacements = [np.linalg.norm(points - positions[0], axis=1)[faces].mean(axis=1) for points in positions]
    scale = matplotlib.colors.Normalize(vmin=0, vmax=max(values.max() for values in displacements))
    cmap = plt.get_cmap("Blues")
    combined = np.concatenate(positions)
    lo, hi = combined.min(axis=0), combined.max(axis=0)
    padding = 0.05 * max(hi - lo)
    lo, hi = lo - padding, hi + padding

    figure, axes = plt.subplots(
        1, 3, figsize=(10, 3.5), subplot_kw={"projection": "3d", "proj_type": "ortho"}, layout="constrained"
    )
    for axis, (_, title), points, values in zip(axes, cases, positions, displacements):
        axis.add_collection3d(Poly3DCollection(
            points[faces], facecolors=cmap(scale(values)), edgecolors="#456476", linewidths=0.5
        ))
        axis.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]),
                 xlabel="x", ylabel="y", zlabel="z", title=title)
        axis.set_box_aspect(hi - lo)
        axis.view_init(elev=24, azim=-60)
    figure.colorbar(matplotlib.cm.ScalarMappable(norm=scale, cmap=cmap), ax=axes,
                    orientation="horizontal", shrink=0.48, label="Mean vertex displacement from rest")

    output = args.output or args.input / "laplacian_optimization.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
