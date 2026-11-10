from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

parser = argparse.ArgumentParser()
parser.add_argument('--uniform', action='store_true', help='Plot the constant-Fp example')
args = parser.parse_args()
p = Path(__file__).resolve().parents[3] / 'output' / ('shell_uniform_fp' if args.uniform else 'schwarz_p_fp_free')
d = np.load(p / "relaxed.npz")
X, Y, F = d["X"], d["Y"], d["faces"]
records = json.loads((p / "summary.json").read_text())
disp = np.linalg.norm(Y - X, axis=1)
fd = disp[F].mean(axis=1)
norm = Normalize(0, disp.max())
cmap = plt.get_cmap("viridis")
lo = np.minimum(X.min(axis=0), Y.min(axis=0))
hi = np.maximum(X.max(axis=0), Y.max(axis=0))
mid = (lo + hi) / 2
radius = max(hi - lo) * 0.55
fig = plt.figure(figsize=(12, 10), layout="constrained")
for j, (elev, azim) in enumerate([(22, 35), (5, -90)]):
    for i, (V, title) in enumerate(
        [(X, "Original"), (Y, "Relaxed under prescribed Fp")]
    ):
        ax = fig.add_subplot(2, 2, 2 * j + i + 1, projection="3d")
        colors = (
            np.tile([0.67, 0.73, 0.80, 1.0], (len(F), 1)) if i == 0 else cmap(norm(fd))
        )
        ax.add_collection3d(
            Poly3DCollection(
                V[F],
                facecolors=colors,
                edgecolors=(0, 0, 0, 0.08),
                linewidths=0.12,
                shade=True,
                lightsource=LightSource(azdeg=45, altdeg=45),
            )
        )
        ax.set(
            xlim=(mid[0] - radius, mid[0] + radius),
            ylim=(mid[1] - radius, mid[1] + radius),
            zlim=(mid[2] - radius, mid[2] + radius),
            title=title,
        )
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev, azim)
        ax.set_axis_off()
fig.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=fig.axes,
    shrink=0.55,
    label="Displacement magnitude (cell length = 1)",
)
fp_label = 'Fp = diag(1, 0.5, 1) on every face' if args.uniform else 'Fp = diag(exp(q), exp(-q), 1), q = 0.65 cos(2 pi Z)'
fig.suptitle('Schwarz P, Koiter-StVK, thickness 0.06: free open boundaries, no PBC, no contact\n' + fp_label + '; actual geometry, no exaggeration')
fig.savefig(p / "comparison.png", dpi=180)
