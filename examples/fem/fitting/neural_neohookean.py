"""Learn a fixed Neo-Hookean material from native energy/stress samples.

Run: uv run --no-sync python examples/fem/fitting/neural_neohookean.py
Writes a checkpoint and independent evaluation report under output/neural_neohookean.
"""

from litepgo.energy import AffineResponse
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
from litepgo import dof, fem, mesh, solver, tbb_thread_limit
from litepgo.torch import TorchVolumetricMaterial


class InvariantMaterial(torch.nn.Module):
    """Isotropic, objective 3-32-32-1 energy; fixed lambda=2 and mu=3.

    W is per original reference volume. Reference energy/stress corrections stay
    differentiable in the weights. This MLP does not guarantee global stability.
    """

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 32),
            torch.nn.Softplus(),
            torch.nn.Linear(32, 32),
            torch.nn.Softplus(),
            torch.nn.Linear(32, 1, bias=False),
        ).double()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Old checkpoints contain an output bias canceled by reference correction.
        state_dict = state_dict.copy()
        state_dict.pop("net.4.bias", None)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    @staticmethod
    def invariants(Fe):
        J = torch.linalg.det(Fe)
        C = Fe.transpose(-1, -2) @ Fe
        I1 = C.diagonal(dim1=-2, dim2=-1).sum(-1)
        I2 = 0.5 * (I1.square() - C.square().sum(dim=(-2, -1)))
        return torch.stack(
            [J.pow(-2 / 3) * I1 - 3, J.pow(-4 / 3) * I2 - 3, J.log()], dim=-1
        )

    def elastic_energy(self, Fe):
        z = self.invariants(Fe)
        # Even value-only inference needs the origin slope; keep it in the graph when training.
        differentiating = torch.is_grad_enabled()
        with torch.inference_mode(False), torch.enable_grad():
            origin = torch.zeros(
                3, dtype=Fe.dtype, device=Fe.device, requires_grad=True
            )
            f0 = self.net(origin).squeeze(-1)
            slope = torch.autograd.grad(f0, origin, create_graph=differentiating)[0][2]
        return self.net(z).squeeze(-1) - f0 - slope * z[..., 2]

    def forward(self, F, Fp, theta):
        if theta.shape != (len(F), 2) or not torch.all(
            theta == theta.new_tensor([2.0, 3.0])
        ):
            raise ValueError("This trained material is fixed at theta=(2,3)")
        Fe = F @ torch.linalg.inv(Fp)
        return torch.linalg.det(Fp) * self.elastic_energy(Fe)


def sample_deformations(count, seed):
    """Independent rotations and uniform principal stretches in [0.7, 1.3]."""
    rng = np.random.default_rng(seed)

    def rotations():
        Q, _ = np.linalg.qr(rng.normal(size=(count, 3, 3)))
        Q[:, :, 2] *= np.linalg.det(Q)[:, None]
        return Q

    stretches = rng.uniform(0.7, 1.3, size=(count, 3))
    return (rotations() * stretches[:, None, :]) @ rotations().transpose(0, 2, 1)


def native_labels(F, Fp=None, tangent=False):
    if Fp is None:
        Fp = np.broadcast_to(np.eye(3), F.shape)
    theta = np.array([2.0, 3.0])
    W = np.array([fem.NeoHookean.eval_W(f, p, theta) for f, p in zip(F, Fp)])
    P = np.array([fem.NeoHookean.eval_dW_dF(f, p, theta) for f, p in zip(F, Fp)])
    if not tangent:
        return torch.tensor(W), torch.tensor(P)
    # Convert both native column-major tangent axes to Torch row-major order.
    order = [0, 3, 6, 1, 4, 7, 2, 5, 8]
    H = np.array(
        [
            fem.NeoHookean.eval_d2W_dF2(f, p, theta)[np.ix_(order, order)]
            for f, p in zip(F, Fp)
        ]
    )
    return torch.tensor(W), torch.tensor(P), torch.tensor(H)


def prediction(model, F, Fp=None, *, training=False, tangent=False):
    F = F.detach().requires_grad_(True)
    if Fp is None:
        Fp = torch.eye(3, dtype=torch.float64).expand(len(F), 3, 3)
    theta = F.new_tensor([2.0, 3.0]).expand(len(F), 2)
    W = model(F, Fp, theta)
    P = torch.autograd.grad(W.sum(), F, create_graph=training or tangent)[0]
    if not tangent:
        return W, P
    H = torch.stack(
        [
            torch.autograd.grad(P.reshape(-1, 9)[:, i].sum(), F, retain_graph=i < 8)[
                0
            ].reshape(-1, 9)
            for i in range(9)
        ],
        dim=1,
    )
    return W, P, H


def normalized_loss(model, F, W, P, scales, training=False):
    w, p = prediction(model, F, training=training)
    return ((w - W) / scales[0]).square().mean() + ((p - P) / scales[1]).square().mean()


def tangent_direction(model, F, direction, *, training=False):
    """Apply the energy Hessian to one direction without forming its matrix."""
    F = F.detach().requires_grad_(True)
    W = model.elastic_energy(F)
    P = torch.autograd.grad(W.sum(), F, create_graph=True)[0]
    return torch.autograd.grad((P * direction).sum(), F, create_graph=training)[0]


def tangent_samples(count, seed):
    F = sample_deformations(count, seed)
    direction = np.random.default_rng(seed + 100).normal(size=F.shape)
    direction /= np.linalg.norm(direction, axis=(1, 2), keepdims=True)
    H = native_labels(F, tangent=True)[2]
    target = torch.einsum("bij,bj->bi", H, torch.tensor(direction.reshape(-1, 9)))
    return torch.tensor(F), torch.tensor(direction), target.reshape(-1, 3, 3)


def tangent_loss(model, samples, scale, *, training=False):
    F, direction, target = samples
    predicted = tangent_direction(model, F, direction, training=training)
    return ((predicted - target) / scale).square().mean()


def train(
    model,
    train_data,
    validation_data,
    *,
    adam_steps=1500,
    lbfgs_steps=300,
    seed=42,
    tangent_supervision=False,
):
    F, W, P = train_data
    scales = (W.square().mean().sqrt(), P.square().mean().sqrt())
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    tangent_train = tangent_samples(128, 21) if tangent_supervision else None
    tangent_validation = tangent_samples(128, 22) if tangent_supervision else None
    tangent_scale = (
        tangent_train[2].square().mean().sqrt() if tangent_supervision else None
    )
    history = []
    best_loss, best_state = float("inf"), None

    def validate(stage, step):
        nonlocal best_loss, best_state
        loss = normalized_loss(model, *validation_data, scales).item()
        if tangent_supervision:
            loss += tangent_loss(model, tangent_validation, tangent_scale).item()
        history.append(dict(stage=stage, step=step, validation_loss=loss))
        if loss < best_loss:
            best_loss, best_state = loss, copy.deepcopy(model.state_dict())
        print(f"{stage} {step}: validation_loss={loss:.6g}", flush=True)

    validate("initial", 0)
    for step in range(adam_steps):
        ids = torch.randint(len(F), (min(512, len(F)),), generator=generator)
        optimizer.zero_grad()
        loss = normalized_loss(model, F[ids], W[ids], P[ids], scales, training=True)
        if tangent_supervision:
            loss = loss + tangent_loss(
                model, tangent_train, tangent_scale, training=True
            )
        loss.backward()
        optimizer.step()
        if (step + 1) % 100 == 0 or step + 1 == adam_steps:
            validate("adam", step + 1)
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=20,
        history_size=100,
        tolerance_grad=1e-10,
        tolerance_change=1e-14,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        loss = normalized_loss(model, F, W, P, scales, training=True)
        if tangent_supervision:
            loss = loss + tangent_loss(
                model, tangent_train, tangent_scale, training=True
            )
        loss.backward()
        return loss

    completed = 0
    while completed < lbfgs_steps:
        iterations = min(20, lbfgs_steps - completed)
        optimizer.param_groups[0]["max_iter"] = iterations
        optimizer.step(closure)
        completed += iterations
        validate("lbfgs", completed)
    model.load_state_dict(best_state)
    return history


def loading_paths():
    stretch = np.linspace(0.7, 1.3, 61)

    def diagonal(entries):
        result = np.zeros((len(stretch), 3, 3))
        result[:, range(3), range(3)] = np.stack(entries, axis=1)
        return result

    ones = np.ones_like(stretch)
    shear = np.broadcast_to(np.eye(3), (61, 3, 3)).copy()
    shear[:, 0, 1] = np.linspace(-0.5, 0.5, 61)
    return {
        "uniaxial": diagonal([stretch, ones, ones]),
        "biaxial": diagonal([stretch, stretch, ones]),
        "volumetric": diagonal([stretch, stretch, stretch]),
        "shear": shear,
    }


def errors(model, F, Fp=None):
    reference = native_labels(F, Fp, tangent=True)
    actual = prediction(
        model, torch.tensor(F), None if Fp is None else torch.tensor(Fp), tangent=True
    )
    report = {}
    for name, a, b in zip(("energy", "stress", "tangent"), actual, reference):
        report[name + "_relative_l2"] = (
            torch.linalg.vector_norm(a.detach() - b) / torch.linalg.vector_norm(b)
        ).item()
        report[name + "_max_absolute"] = (a.detach() - b).abs().max().item()
    return report


def fem_material(model):
    """A Python material instance for native FEM; the trained model stays fixed."""

    class LearnedNeoHookean(TorchVolumetricMaterial):
        num_parameters = 2
        def __init__(self, module):
            super().__init__()
            self.module = module

        def W(self, F, Fp, theta):
            return self.module(F, Fp, theta)

    return LearnedNeoHookean(model)


def equilibrium_errors(model):
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    topology = mesh.TetTopology(
        5, np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32)
    )
    mapping = dof.VertexDofMap(topology, 3)
    Field = fem.InterpolatedResponse[fem.TetP1Basis]
    geometry = [Field(mapping.gather(e, nodes.ravel())) for e in range(2)]
    theta = [AffineResponse(np.zeros((2, 3)), np.array([2.0, 3.0]))] * 2
    fixed = dof.DofReduction(15, [*range(9), 12, 13, 14])
    Native = fem.VolumetricFemEnergy[fem.TetP1Basis, fem.TetCentroidQuadrature]
    Learned = fem.VolumetricFemEnergy[fem.TetP1Basis, fem.TetCentroidQuadrature]
    reports = []
    for plastic in (
        np.eye(3),
        np.array([[1.04, 0.03, 0.0], [0.0, 0.98, 0.02], [0.01, 0.0, 1.02]]),
    ):
        Fp = [AffineResponse(np.zeros((9, 3)), plastic.ravel(order="F"))] * 2
        learned = Learned(fem_material(model), mapping, geometry, Fp, theta)
        native = Native(fem.PointwiseVolumetricAdapter[fem.NeoHookean](), mapping, geometry, Fp, theta)
        for load in ([0.04, -0.02, 0.06], [-0.03, 0.02, -0.04], [0.1, 0.04, 0.08]):
            u = np.zeros(15)
            u[12:] = load
            options = solver.NewtonOptions(gradient_tolerance=1e-10, max_iterations=100)
            a, b = [
                solver.newton_solve(E, fixed, u, options) for E in (learned, native)
            ]
            converged = a.status == b.status == solver.NewtonStatus.Converged
            if not converged:
                raise RuntimeError("Neural/native equilibrium failed")
            relative = np.linalg.norm(
                a.x[fixed.free_dofs] - b.x[fixed.free_dofs]
            ) / np.linalg.norm(b.x[fixed.free_dofs])
            force_error = np.linalg.norm(
                learned.evaluate(a.x).gradient()[fixed.fixed_dofs]
                - native.evaluate(b.x).gradient()[fixed.fixed_dofs]
            ) / np.linalg.norm(native.evaluate(b.x).gradient()[fixed.fixed_dofs])
            stretches = []
            for e, X in enumerate(geometry):
                field = Field(mapping.gather(e, a.x))
                F = (
                    fem.DeformationGradientResponse(field, X).evaluate(np.full(3, 0.25)).value.reshape(3, 3, order='F')
                )
                stretches.extend(
                    np.linalg.svd(F @ np.linalg.inv(plastic), compute_uv=False)
                )
            reports.append(
                dict(
                    load=load,
                    Fp=plastic.tolist(),
                    converged=bool(converged),
                    free_displacement_relative_error=float(relative),
                    reaction_relative_error=float(force_error),
                    elastic_stretch_range=[
                        float(min(stretches)),
                        float(max(stretches)),
                    ],
                )
            )
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/neural_neohookean"))
    parser.add_argument("--adam-steps", type=int, default=1500)
    parser.add_argument("--lbfgs-steps", type=int, default=300)
    parser.add_argument(
        "--tangent-supervision",
        action="store_true",
        help="Add 128 directional tangent labels to the training loss",
    )
    args = parser.parse_args()
    if args.adam_steps < 0 or args.lbfgs_steps < 0:
        parser.error("step counts must be nonnegative")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = InvariantMaterial()
    start = time.perf_counter()
    datasets = []
    for count, seed in ((10000, 11), (2000, 12)):
        F = sample_deformations(count, seed)
        datasets.append((torch.tensor(F), *native_labels(F)))
    history = train(
        model,
        *datasets,
        adam_steps=args.adam_steps,
        lbfgs_steps=args.lbfgs_steps,
        tangent_supervision=args.tangent_supervision,
    )
    model.eval().requires_grad_(False)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "theta": [2.0, 3.0], "seed": 42},
        args.output / "material.pt",
    )
    report = {
        "seed": 42,
        "theta": [2.0, 3.0],
        "train_samples": 10000,
        "validation_samples": 2000,
        "adam_steps": args.adam_steps,
        "lbfgs_steps": args.lbfgs_steps,
        "history": history,
        "tangent_supervision": args.tangent_supervision,
        "tangent_samples": 128 if args.tangent_supervision else 0,
    }
    report["random_test"] = errors(model, sample_deformations(512, 13))
    report["paths"] = {name: errors(model, F) for name, F in loading_paths().items()}
    Fe = sample_deformations(256, 14)
    Fp = np.broadcast_to(
        np.array([[1.04, 0.03, 0.0], [0.0, 0.98, 0.02], [0.01, 0.0, 1.02]]), Fe.shape
    ).copy()
    report["plastic_test"] = errors(model, Fe @ Fp, Fp)
    with tbb_thread_limit(1):
        report["equilibrium"] = equilibrium_errors(model)
    # Pilot acceptance criteria use aggregate errors, not pointwise relative errors near zero.
    limits = {
        "energy_relative_l2": 0.01,
        "stress_relative_l2": 0.01,
        "tangent_relative_l2": 0.02,
    }
    report["acceptance_limits"] = {**limits, "equilibrium_relative_error": 0.01}
    material_checks = [
        report["random_test"],
        report["plastic_test"],
        *report["paths"].values(),
    ]
    report["passed"] = all(
        case[key] < limit for case in material_checks for key, limit in limits.items()
    ) and all(
        case["free_displacement_relative_error"] < 0.01
        and case["reaction_relative_error"] < 0.01
        and 0.7
        <= case["elastic_stretch_range"][0]
        <= case["elastic_stretch_range"][1]
        <= 1.3
        for case in report["equilibrium"]
    )
    report["seconds"] = time.perf_counter() - start
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
