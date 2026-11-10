"""Compare raw freely relaxed shapes; no post-solve alignment or exaggeration."""

from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

p = Path(__file__).resolve().parents[3] / "output/shell_interpolated_fp"
q = np.load(p / "q1.npz")
h = np.load(p / "hermite.npz")
X = q["X"]
faces = q["faces"]
vertices = [X, q["Y"], h["Y"]]
names = ["Original", "Q1 plastic displacement", "Hermite plastic displacement"]
disps = [np.linalg.norm(Y - X, axis=1) for Y in vertices]
norm = Normalize(0, max(d.max() for d in disps))
allv = np.concatenate(vertices)
mid = (allv.min(0) + allv.max(0)) / 2
radius = np.ptp(allv, axis=0).max() * 0.53
fig = plt.figure(figsize=(15, 9), layout="constrained")
for row, view in enumerate([(23, 35), (0, -90)]):
    for col, (Y, title, d) in enumerate(zip(vertices, names, disps)):
        ax = fig.add_subplot(2, 3, row * 3 + col + 1, projection="3d")
        colors = (
            np.tile([0.7, 0.75, 0.8, 1], (len(faces), 1))
            if col == 0
            else plt.cm.viridis(norm(d[faces].mean(1)))
        )
        ax.add_collection3d(
            Poly3DCollection(
                Y[faces],
                facecolors=colors,
                edgecolors=(0, 0, 0, 0.08),
                linewidths=0.1,
                shade=True,
                lightsource=LightSource(45, 45),
            )
        )
        ax.set(
            xlim=(mid[0] - radius, mid[0] + radius),
            ylim=(mid[1] - radius, mid[1] + radius),
            zlim=(mid[2] - radius, mid[2] + radius),
            title=title,
        )
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(*view)
        ax.set_axis_off()
fig.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap="viridis"),
    ax=fig.axes,
    shrink=0.55,
    label="Displacement magnitude (cell length = 1)",
)
fig.suptitle(
    "Schwarz P, Koiter-StVK: Fp sampled from a 60-degree plastic twist\nRest-start minimization; all vertices free; no contact or PBC"
)
fig.savefig(p / "comparison.png", dpi=180)
