"""Plot energy and force ratios for the three-group contact example."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV from contact_pair_weights")
    parser.add_argument("-o", "--output", type=Path, help="PNG path (default: input stem)")
    args = parser.parse_args()
    data = np.atleast_1d(np.genfromtxt(args.input, delimiter=",", names=True))
    expected = ("pair", "group_a", "group_b", "multiplier", "reference_energy", "weighted_energy",
                "reference_force", "weighted_force", "active_records")
    if (data.dtype.names != expected or len(data) != 3 or not np.isfinite(data.tolist()).all()
            or np.any(data["reference_energy"] <= 0) or np.any(data["reference_force"] <= 0)):
        parser.error("expected the example's three finite pair samples with positive reference energy and force")
    figure, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    figure.suptitle("Primitive group multipliers\nOne mesh, three isolated contacts · area weighting on")
    x = np.arange(len(data))
    labels = [f"{int(row['group_a'])} ↔ {int(row['group_b'])}" for row in data]
    for axis, key, title in zip(axes, ("energy", "force"), ("Contact energy", "Normal force")):
        ratio = data[f"weighted_{key}"] / data[f"reference_{key}"]
        if not np.allclose(ratio, data["multiplier"], rtol=1e-10, atol=1e-12):
            parser.error("energy and force ratios must agree with the group multipliers")
        axis.bar(x, ratio, color=["#197b91", "#cc7900", "#8a9299"], width=.55)
        axis.scatter(x, data["multiplier"], marker="_", s=450, color="black", label="Requested multiplier", zorder=3)
        axis.axhline(1, color="gray", linestyle="--", linewidth=1, label="Without group multipliers")
        for i, (value, records) in enumerate(zip(ratio, data["active_records"])):
            axis.text(i, value + .08, f"{value:g}x\n{int(records)} active record", ha="center", fontsize=10)
        axis.set(xticks=x, xticklabels=labels, xlabel="Primitive group pair", ylabel="Relative to the same unscaled contact",
                 title=title, ylim=(0, 2.65))
        axis.grid(axis="y", alpha=.15)
    axes[0].legend(loc="upper left", fontsize=7)
    output = args.output or args.input.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
