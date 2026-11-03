"""Plot the deformed VEG and matching boundary OBJ from the plate compression example."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from litepgo import mesh

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="deformed VEG with a same-stem OBJ")
    parser.add_argument("-o", "--output", type=Path, help="PNG path, defaulting to the input stem")
    args = parser.parse_args()

    volume = mesh.read_veg(args.input)
    surface = mesh.read_obj(args.input.with_suffix(".obj"))
    positions, faces = surface.positions, surface.cells
    starts = faces.ravel()
    ends = faces[:, [1, 2, 0]].ravel()
    graph = coo_matrix((np.ones(len(starts)), (starts, ends)), shape=(len(positions), len(positions)))
    count, labels = connected_components(graph, directed=False)
    if count != 2:
        parser.error("expected two connected components in the boundary OBJ")

    lower, upper = sorted(range(count), key=lambda group: positions[labels == group, 2].mean())
    colors = ("#176b87", "#d47a24")
    figure = plt.figure(figsize=(10, 5), layout="constrained")
    figure.suptitle(f"Plate compression equilibrium · {volume.topology.num_cells:,} tetrahedra")
    view = figure.add_subplot(1, 2, 1, projection="3d")
    detail = figure.add_subplot(1, 2, 2)
    for group, color in zip((lower, upper), colors):
        triangles = positions[faces[labels[faces[:, 0]] == group]]
        view.add_collection3d(Poly3DCollection(triangles, facecolor=color, edgecolor="white", linewidth=0.25))
        projected = triangles[:, :, [0, 2]]
        segments = projected[:, [[0, 1], [1, 2], [2, 0]], :].reshape(-1, 2, 2)
        detail.add_collection(LineCollection(segments, colors=color, linewidths=0.55, alpha=0.5))

    lo, hi = positions.min(axis=0), positions.max(axis=0)
    view.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]), xlabel="x", ylabel="y", zlabel="z")
    view.set_box_aspect(hi - lo)
    view.view_init(elev=22, azim=-55)
    view.set_title("Deformed boundary")

    lower_top = positions[labels == lower, 2].max()
    upper_bottom = positions[labels == upper, 2].min()
    padding = 0.12 * (hi[2] - lo[2])
    detail.set(xlim=(lo[0], hi[0]), ylim=(min(lower_top, upper_bottom) - padding, max(lower_top, upper_bottom) + padding),
               xlabel="x", ylabel="z", title="Contact region · x–z projection")
    detail.set_aspect("equal", adjustable="box")

    output = args.output or args.input.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
