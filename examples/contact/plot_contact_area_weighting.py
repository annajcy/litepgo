"""Plot relative mesh sensitivity from the contact_area_weighting CSV."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV from contact_area_weighting")
    parser.add_argument("-o", "--output", type=Path, help="PNG path, defaulting to the input stem")
    args = parser.parse_args()
    data = np.atleast_1d(np.genfromtxt(args.input, delimiter=",", names=True))
    required = ("subdivisions", "edge_length", "triangles", "use_area_weighting", "energy", "normal_force")
    if data.dtype.names != required:
        parser.error("expected the contact_area_weighting CSV columns")

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    figure.suptitle("Area weighting under mesh refinement\nFixed sheets: side=0.16, gap=0.001, dhat=0.002, kappa=1e5")
    for weighted, color in ((False, "#d47a24"), (True, "#176b87")):
        rows = np.sort(data[data["use_area_weighting"] == int(weighted)], order="subdivisions")
        if (len(rows) < 2 or not np.isfinite(rows.tolist()).all()
                or any(np.any(rows[key] <= 0) for key in required if key != "use_area_weighting")):
            parser.error("expected at least two finite positive mesh samples per mode")
        for axis, key, title in zip(axes, ("energy", "normal_force"), ("Contact energy", "Total normal force")):
            relative = rows[key] / rows[key][0]
            label = "Area weighting on" if weighted else "Area weighting off"
            axis.plot(rows["subdivisions"], relative, "o-", color=color, label=label)
            axis.annotate(f"{relative[-1]:.2f}x", (rows["subdivisions"][-1], relative[-1]),
                          xytext=(-5, 7), textcoords="offset points", ha="right", color=color)
            axis.set(xlabel="Grid subdivisions per edge", ylabel="Relative to each mode's coarsest mesh", title=title)
            axis.set_xscale("log", base=2)
            axis.set_xticks(rows["subdivisions"], rows["subdivisions"].astype(int))
            axis.grid(alpha=.2)
    for axis in axes:
        axis.set_ylim(0, 1.15 * max(line.get_ydata().max() for line in axis.lines))
    axes[0].legend()
    output = args.output or args.input.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
