from pathlib import Path
import runpy

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def example():
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/fem/fitting/neural_neohookean.py"
    )
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield runpy.run_path(str(path))
    torch.set_num_threads(previous)


def make_model(example):
    with torch.random.fork_rng():
        torch.manual_seed(42)
        return example["InvariantMaterial"]()


def test_reference_state_and_isotropy_by_construction(example):
    model = make_model(example)
    W, P, H = example["prediction"](
        model, torch.eye(3, dtype=torch.float64)[None], tangent=True
    )
    assert abs(W.item()) < 1e-13
    assert P.abs().max().item() < 1e-13
    assert torch.isfinite(H).all()
    F = torch.tensor(example["sample_deformations"](12, 71))
    Q, _ = torch.linalg.qr(torch.tensor(np.random.default_rng(6).normal(size=(3, 3))))
    Q[:, 2] *= torch.linalg.det(Q)
    W, P = example["prediction"](model, F)
    rotated_W, rotated_P = example["prediction"](model, Q @ F)
    torch.testing.assert_close(rotated_W, W, atol=1e-13, rtol=1e-11)
    torch.testing.assert_close(rotated_P, Q @ P, atol=1e-13, rtol=1e-11)
    torch.testing.assert_close(
        example["prediction"](model, F @ Q)[0], W, atol=1e-13, rtol=1e-11
    )
    with torch.no_grad():
        torch.testing.assert_close(model.elastic_energy(F), W)
    with torch.inference_mode():
        torch.testing.assert_close(model.elastic_energy(F), W)


def test_stress_loss_retains_weight_gradient_through_reference_correction(example):
    model = make_model(example)
    F = torch.tensor(example["sample_deformations"](8, 16))
    W, P = example["native_labels"](F.numpy())
    scales = (W.square().mean().sqrt(), P.square().mean().sqrt())
    loss = example["normalized_loss"](model, F, W, P, scales, training=True)
    loss.backward()
    parameter = model.net[0].weight
    # The volume input participates in the reference-pressure correction.
    index = (3, 2)
    analytic = parameter.grad[index].item()
    original = parameter[index].item()
    h = 1e-5
    values = []
    for offset in (h, -h):
        with torch.no_grad():
            parameter[index] = original + offset
        values.append(example["normalized_loss"](model, F, W, P, scales).item())
    with torch.no_grad():
        parameter[index] = original
    assert analytic == pytest.approx(
        (values[0] - values[1]) / (2 * h), rel=2e-5, abs=1e-8
    )


def test_plastic_volume_factor_and_checkpoint_roundtrip(example, tmp_path):
    model = make_model(example)
    Fe = torch.tensor(example["sample_deformations"](5, 34))
    Fp = Fe.new_tensor(
        [[1.04, 0.03, 0.0], [0.0, 0.98, 0.02], [0.01, 0.0, 1.02]]
    ).expand(5, 3, 3)
    theta = Fe.new_tensor([2.0, 3.0]).expand(5, 2)
    torch.testing.assert_close(
        model(Fe @ Fp, Fp, theta), torch.linalg.det(Fp) * model.elastic_energy(Fe)
    )
    path = tmp_path / "model.pt"
    torch.save({"state_dict": model.state_dict()}, path)
    restored = example["InvariantMaterial"]()
    restored.load_state_dict(torch.load(path, weights_only=True)["state_dict"])
    restored.requires_grad_(False)
    a = example["prediction"](model, Fe @ Fp, Fp, tangent=True)
    b = example["prediction"](restored, Fe @ Fp, Fp, tangent=True)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y)
    with pytest.raises(ValueError, match="fixed"):
        model(Fe, Fp, theta + 1)


@pytest.mark.parametrize("tangent_supervision", [False, True])
def test_small_training_run_and_native_tangent_order(example, tangent_supervision):
    model = make_model(example)
    data = []
    for n, seed in ((128, 11), (64, 12)):
        F = example["sample_deformations"](n, seed)
        data.append((torch.tensor(F), *example["native_labels"](F)))
    history = example["train"](
        model,
        *data,
        adam_steps=20,
        lbfgs_steps=20,
        tangent_supervision=tangent_supervision,
    )
    assert (
        min(row["validation_loss"] for row in history)
        < 0.1 * history[0]["validation_loss"]
    )
    F = example["sample_deformations"](4, 13)
    _, _, H = example["native_labels"](F, tangent=True)
    direction = np.random.default_rng(9).normal(size=F.shape)
    h = 1e-6
    plus = example["native_labels"](F + h * direction)[1]
    minus = example["native_labels"](F - h * direction)[1]
    np.testing.assert_allclose(
        np.einsum("bij,bj->bi", H.numpy(), direction.reshape(-1, 9)),
        ((plus - minus) / (2 * h)).numpy().reshape(-1, 9),
        rtol=1e-7,
        atol=1e-8,
    )
    for path in example["loading_paths"]().values():
        stretch = np.linalg.svd(path, compute_uv=False)
        assert stretch.min() >= 0.7 - 1e-12 and stretch.max() <= 1.3 + 1e-12


def test_directional_tangent_and_weight_gradient(example):
    model = make_model(example)
    F, direction, target = example["tangent_samples"](5, 21)
    product = example["tangent_direction"](model, F, direction)
    H = example["prediction"](model, F, tangent=True)[2]
    expected = torch.einsum("bij,bj->bi", H, direction.reshape(-1, 9))
    torch.testing.assert_close(product.reshape(-1, 9), expected)
    scale = target.square().mean().sqrt()
    loss = example["tangent_loss"](model, (F, direction, target), scale, training=True)
    loss.backward()
    parameter = model.net[0].weight
    index = (3, 2)
    analytic = parameter.grad[index].item()
    original = parameter[index].item()
    values = []
    for offset in (1e-5, -1e-5):
        with torch.no_grad():
            parameter[index] = original + offset
        values.append(
            example["tangent_loss"](model, (F, direction, target), scale).item()
        )
    with torch.no_grad():
        parameter[index] = original
    assert analytic == pytest.approx((values[0] - values[1]) / 2e-5, rel=2e-5, abs=1e-8)


def test_legacy_output_bias_is_ignored(example):
    model = make_model(example)
    state = model.state_dict()
    state["net.4.bias"] = torch.tensor([12.0], dtype=torch.float64)
    restored = make_model(example)
    restored.load_state_dict(state)
    F = torch.tensor(example["sample_deformations"](3, 7))
    torch.testing.assert_close(model.elastic_energy(F), restored.elastic_energy(F))
