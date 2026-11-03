"""Plot the rest and edited OBJ meshes from the ARAP example."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

from litepgo import mesh

from volume_mesh import end_faces

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="directory containing rest.obj and edited.obj")
    parser.add_argument("-o", "--output", type=Path, help="PNG path, defaulting to arap.png in the input directory")
    args = parser.parse_args()

    rest = mesh.read_obj(args.input / "rest.obj")
    edited = mesh.read_obj(args.input / "edited.obj")
    faces = rest.cells
    if not len(faces) or rest.positions.shape != edited.positions.shape or not np.array_equal(faces, edited.cells):
        parser.error("rest.obj and edited.obj must have the same nonempty topology and vertex ordering")

    original, result = rest.positions, edited.positions
    left, right = end_faces(original)
    lo = np.minimum(original.min(axis=0), result.min(axis=0))
    hi = np.maximum(original.max(axis=0), result.max(axis=0))
    padding = 0.05 * max(hi - lo)
    lo, hi = lo - padding, hi + padding

    figure = plt.figure(figsize=(10, 5), layout="constrained")
    figure.suptitle("ARAP deformation")
    for index, (title, positions) in enumerate((("Rest", original), ("Edited", result)), 1):
        axis = figure.add_subplot(1, 2, index, projection="3d", proj_type="ortho")
        axis.add_collection3d(Poly3DCollection(
            positions[faces], facecolor="#dbeaf0", edgecolor="#456476", linewidth=0.5
        ))
        axis.scatter(*positions[left].T, color="#176b87", label="Fixed end")
        axis.scatter(*positions[right].T, color="#d47a24", label="Moved end")
        axis.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]),
                 xlabel="x", ylabel="y", zlabel="z", title=title)
        axis.set_box_aspect(hi - lo)
        axis.view_init(elev=26, azim=-63)
    axis.legend(loc="upper right")
    output = args.output or args.input / "arap.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
