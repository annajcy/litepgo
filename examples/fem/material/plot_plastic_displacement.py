"""Visualize the prescribed displacement, before shell relaxation."""

from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from litepgo import fem
from shell_interpolated_fp import plastic_fields

root = Path(__file__).resolve().parents[3]
p = root / "output/shell_interpolated_fp"
d = np.load(p / "q1.npz")
X, faces = d["X"], d["faces"]
lo, hi = X.min(0), X.max(0)
size = hi - lo
corners = np.array([fem.HexQ1Basis.coefficient_info(i).point for i in range(8)])
world = lo + corners * size
edges = [
    (i, j)
    for i in range(8)
    for j in range(i + 1, 8)
    if np.count_nonzero(corners[i] != corners[j]) == 1
]
fig = plt.figure(figsize=(15, 10), layout="constrained")
for col, (name, basis) in enumerate(
    [("Q1", fem.HexQ1Basis), ("Hermite", fem.HexHermiteBasis)]
):
    _, up, _, _ = plastic_fields(basis, lo, hi)

    def displace(xi):
        return np.array([up.evaluate(v).value for v in xi])

    uc = displace(corners)
    Y = world + uc
    grid = np.array(
        [
            [x, y, z]
            for z in np.linspace(0, 1, 5)
            for y in np.linspace(0, 1, 4)
            for x in np.linspace(0, 1, 4)
        ]
    )
    G = lo + grid * size
    ug = displace(grid)
    for row in range(2):
        ax = fig.add_subplot(2, 2, 2 * row + col + 1, projection="3d")
        for i, j in edges:
            xi = (1 - np.linspace(0, 1, 41)[:, None]) * corners[i] + np.linspace(
                0, 1, 41
            )[:, None] * corners[j]
            curve = lo + xi * size + displace(xi)
            ax.plot(*world[[i, j]].T, c="gray", ls="--", lw=1)
            ax.plot(*curve.T, c="tab:orange", lw=2)
        if row == 0:
            ax.scatter(*world.T, c="black", s=20)
            ax.scatter(*Y.T, c="tab:orange", s=25)
            ax.quiver(
                *world.T,
                *uc.T,
                color="tab:blue",
                normalize=False,
                arrow_length_ratio=0.15,
            )
            for i, pos in enumerate(world):
                ax.text(*pos, str(i), fontsize=8)
            ax.add_collection3d(
                Poly3DCollection(
                    X[faces], facecolors=(0.6, 0.65, 0.7, 0.12), edgecolors="none"
                )
            )
            ax.set_title(name + ": corner up and deformed bounding-box edges")
        else:
            ax.quiver(
                *G.T,
                *ug.T,
                color="tab:blue",
                normalize=False,
                arrow_length_ratio=0.15,
                linewidth=0.7,
            )
            ax.set_title(name + ": interpolated up at interior sample points")
        ax.set(
            xlim=(lo[0] - 0.25 * size[0], hi[0] + 0.25 * size[0]),
            ylim=(lo[1] - 0.25 * size[1], hi[1] + 0.25 * size[1]),
            zlim=(lo[2] - 0.1 * size[2], hi[2] + 0.1 * size[2]),
            xlabel="X",
            ylabel="Y",
            zlabel="Z",
        )
        ax.set_box_aspect((1.5, 1.5, 1.2))
        ax.view_init(23, 35)
fig.suptitle(
    "Prescribed plastic displacement (not the relaxed shell)\nGray dashed: original box; orange: X + interpolated up on box edges; blue: up, scale 1:1"
)
fig.savefig(p / "plastic_displacement.png", dpi=180)
