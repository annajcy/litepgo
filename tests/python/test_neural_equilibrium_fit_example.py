"""Network weights -> constitutive mixed derivatives -> equilibrium loss."""

import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import runpy

import numpy as np
import pytest
import torch

from litepgo import tbb_thread_limit


@pytest.fixture(scope="module")
def example():
    directory = Path(__file__).resolve().parents[2] / "examples/fem/fitting"
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with pytest.MonkeyPatch.context() as patch, tbb_thread_limit(1):
            patch.syspath_prepend(str(directory))
            yield runpy.run_path(str(directory / "neural_equilibrium_fit.py"))
    finally:
        torch.set_num_threads(threads)


def test_weight_mapping_and_material_mixed_derivatives(example):
    material, model, unpack, reference = example["make_material"]()
    saved = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(ValueError, match="2 final-layer weights"):
        unpack(torch.zeros(3))
    F = np.array([[[1.08, .04, 0], [.01, .96, .03], [0, .02, 1.05]],
                  [[.97, 0, .02], [.03, 1.06, 0], [0, .01, 1.02]]])
    Fp = np.broadcast_to(np.eye(3), F.shape).copy()
    theta = np.stack([reference.numpy(), reference.numpy() + [.12, -.08]])
    outputs = ("W", "dW_dF", "d2W_dF_dtheta")
    actual = material.eval(F, Fp, theta, outputs=outputs)
    for n in range(2):
        # Independently load the weights into a copy; production never mutates model.
        loaded = copy.deepcopy(model)
        with torch.no_grad():
            loaded.net[4].weight[0, :2].copy_(torch.from_numpy(theta[n]))
        f, fp = torch.from_numpy(F[n:n + 1]), torch.from_numpy(Fp[n:n + 1])
        physical = f.new_tensor([[2., 3.]])
        expected = loaded(f, fp, physical) + example["TorchNeoHookean"].W(f, fp, physical)
        np.testing.assert_allclose(actual["W"][n], expected.item(), atol=1e-14)
        single = material.eval(F[n:n + 1], Fp[n:n + 1], theta[n:n + 1], outputs=outputs)
        for name in outputs:
            np.testing.assert_allclose(single[name][0], actual[name][n], atol=1e-13)
    for j in range(2):
        delta = np.zeros_like(theta)
        delta[:, j] = 1e-4
        plus = material.eval(F, Fp, theta + delta, outputs=("dW_dF",))["dW_dF"]
        minus = material.eval(F, Fp, theta - delta, outputs=("dW_dF",))["dW_dF"]
        np.testing.assert_allclose(actual["d2W_dF_dtheta"][..., j], (plus - minus) / 2e-4,
                                   atol=1e-10, rtol=1e-6)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, saved[name], rtol=0, atol=0)


def test_shared_network_calls_are_thread_safe(example):
    material, _, _, reference = example["make_material"]()
    F = np.array([[[1.08, .04, 0], [.01, .96, .03], [0, .02, 1.05]]])
    Fp = np.eye(3)[None]
    parameters = [reference.numpy()[None] + offset for offset in np.linspace(-.2, .2, 12)]

    def evaluate(theta):
        return material.eval(F, Fp, theta, outputs=("W", "d2W_dF_dtheta"))

    expected = [evaluate(theta) for theta in parameters]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(evaluate, parameters))
    for a, b in zip(actual, expected):
        for name in b:
            np.testing.assert_allclose(a[name], b[name], rtol=1e-10, atol=1e-12)


@pytest.fixture(scope="module")
def problem(example):
    return example["make_fit"]()


def test_equilibrium_gradient_and_batch_partition(example, problem):
    equilibrium, initial, reference, loss = problem
    p = initial.clone().requires_grad_()
    gradient = torch.autograd.grad(loss(p), p)[0]
    assert torch.all(gradient.abs() > 1e-6)
    # Both step sizes re-solve equilibrium; they do not differentiate a saved state.
    for h in (1e-3, 3e-4):
        numerical = []
        for delta in h * torch.eye(2, dtype=torch.float64):
            numerical.append((loss(initial + delta) - loss(initial - delta)).item() / (2 * h))
        np.testing.assert_allclose(gradient.numpy(), numerical, rtol=2e-4, atol=2e-6)

    material, _, _, _ = example["make_material"]()
    _, free, make_energy, make_response = example["make_equilibrium"](material)
    for weights in (reference.numpy(), initial.numpy()):
        u = equilibrium.eval_value(weights)
        potential = make_energy(weights)
        np.testing.assert_allclose(potential.eval_gradient(u)[free], 0, atol=1e-11)
        K = potential.eval_hessian(u).toarray()[np.ix_(free, free)]
        assert np.linalg.eigvalsh(K).min() > 0
        partial = make_response(u).eval_jacobian(weights).toarray()
        for j, delta in enumerate(1e-4 * np.eye(2)):
            fd = (make_energy(weights + delta).eval_gradient(u)
                  - make_energy(weights - delta).eval_gradient(u)) / 2e-4
            np.testing.assert_allclose(partial[:, j], fd, atol=1e-10, rtol=1e-6)

    single, _, _, single_loss = example["make_fit"](batch_size=1)
    np.testing.assert_allclose(single.eval_value(initial.numpy()), equilibrium.eval_value(initial.numpy()), atol=1e-11)
    q = initial.clone().requires_grad_()
    torch.testing.assert_close(single_loss(q), loss(p), atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(torch.autograd.grad(single_loss(q), q)[0], gradient, atol=1e-8, rtol=1e-7)


def test_optimization_reduces_displacement_loss(example, problem):
    _, initial, _, loss = problem
    fitted, before, after = example["fit"](initial, loss)
    assert before == pytest.approx(0.5)
    assert torch.isfinite(fitted).all()
    assert after < before * 1e-6


@pytest.mark.parametrize("width", [2, 8, 32])
def test_batched_material_matches_functional_reference(example, width):
    from litepgo.torch import TorchVolumetricMaterial

    material, model, unpack, reference = example["make_material"](width)

    class Pointwise(TorchVolumetricMaterial):
        num_parameters = width

        @staticmethod
        def W(F, Fp, theta):
            physical = F.new_tensor([2., 3.]).expand(len(F), 2)
            values = [torch.func.functional_call(model, unpack(theta[i]),
                      (F[i:i+1], Fp[i:i+1], physical[i:i+1]))[0] for i in range(len(F))]
            return torch.stack(values) + example["TorchNeoHookean"].W(F, Fp, physical)

    rng = np.random.default_rng(24)
    F = np.eye(3)[None] + .04 * rng.normal(size=(3, 3, 3))
    Fp = np.eye(3)[None] + .02 * rng.normal(size=(3, 3, 3))
    theta = reference.numpy()[None] + .1 * rng.normal(size=(3, width))
    outputs = ("W", "dW_dF", "d2W_dF2", "d2W_dF_dtheta", "d2W_dF_dFp")
    expected = Pointwise.eval(F, Fp, theta, outputs=outputs)
    actual = material.eval(F, Fp, theta, outputs=outputs)
    for name in outputs:
        np.testing.assert_allclose(actual[name], expected[name], rtol=1e-9, atol=1e-11)
