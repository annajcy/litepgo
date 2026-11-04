"""Plot native and learned energy, stress, and tangent on independent loading paths.

Run: uv run --no-sync python examples/fem/fitting/plot_neural_neohookean.py
Requires the project's plot dependency group; run neural_neohookean.py first.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from neural_neohookean import (
    InvariantMaterial,
    loading_paths,
    native_labels,
    prediction,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("output/neural_neohookean/material.pt"),
    )
    parser.add_argument(
        "--baseline", type=Path, help="Optional energy/stress-only checkpoint"
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    model = InvariantMaterial()
    model.load_state_dict(
        torch.load(args.input, map_location="cpu", weights_only=True)["state_dict"]
    )
    model.eval().requires_grad_(False)
    baseline = None
    if args.baseline:
        baseline = InvariantMaterial()
        baseline.load_state_dict(
            torch.load(args.baseline, map_location="cpu", weights_only=True)[
                "state_dict"
            ]
        )
        baseline.eval().requires_grad_(False)
    fig, axes = plt.subplots(3, 4, figsize=(14, 8), layout="constrained")
    for col, (name, F) in enumerate(loading_paths().items()):
        reference = native_labels(F, tangent=True)
        actual = prediction(model, torch.tensor(F), tangent=True)
        shear = name == "shear"
        x = F[:, 0, 1] if shear else F[:, 0, 0]
        axis = 1 if shear else 0
        a = [actual[0], actual[1].reshape(-1, 9)[:, axis], actual[2][:, axis, axis]]
        b = [
            reference[0],
            reference[1].reshape(-1, 9)[:, axis],
            reference[2][:, axis, axis],
        ]
        baseline_values = None
        if baseline is not None:
            bw, bp, bh = prediction(baseline, torch.tensor(F), tangent=True)
            baseline_values = [bw, bp.reshape(-1, 9)[:, axis], bh[:, axis, axis]]
        for row, (y, truth) in enumerate(zip(a, b)):
            ax = axes[row, col]
            ax.plot(x, truth.numpy(), color="#202f45", lw=2, label="Native Neo-Hookean")
            ax.plot(
                x,
                y.detach().numpy(),
                color="#e97824",
                ls="--",
                lw=1.8,
                label="With tangent labels"
                if baseline is not None
                else "Learned energy network",
            )
            if baseline_values is not None:
                ax.plot(
                    x,
                    baseline_values[row].detach().numpy(),
                    color="#7b61a8",
                    ls=":",
                    lw=1.5,
                    label="Energy/stress only",
                )
            ax.grid(alpha=0.2)
            ax.set_ylabel(
                ["Energy W", "P12" if shear else "P11", "A1212" if shear else "A1111"][
                    row
                ]
            )
            if row == 0:
                ax.set_title(name.capitalize())
            if row == 2:
                ax.set_xlabel("Shear γ" if shear else "Stretch s")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Neural material: independent loading paths • λ=2, μ=3", fontsize=15)
    output = args.input.with_name("comparison.png")
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
