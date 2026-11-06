"""CPU float64 PyTorch adapters, constitutive FEM energy, and force responses.

TorchVolumetricFemEnergy assembles a Torch density through native MeshEnergy;
its material/Fp responses provide explicit Jacobians and direct JVP/VJP.
The energy/response functions below are first-order outer autograd adapters.

Only the explicit tensor input is differentiable. Callbacks and model configuration
must remain fixed from forward through differentiation. Response products are
evaluated at a saved input copy; no response Jacobian is saved.
"""

import numpy as np
import torch
from torch.autograd.function import once_differentiable

from litepgo.torch_fem import (
    TorchVolumetricFemEnergy,
    TorchVolumetricFemMaterialResponse,
    TorchVolumetricFemFpResponse,
)

__all__ = [
    "energy",
    "response",
    "TorchVolumetricFemEnergy",
    "TorchVolumetricFemMaterialResponse",
    "TorchVolumetricFemFpResponse",
]


def _input(x):
    if not isinstance(x, torch.Tensor):
        raise TypeError("expected a torch.Tensor")
    if x.device.type != "cpu" or x.dtype != torch.float64 or x.ndim != 1:
        raise ValueError("expected a one-dimensional CPU float64 tensor")
    return x.detach().numpy().copy()


class _Energy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, potential, need_gradient):
        values = _input(x)
        if need_gradient:
            value, gradient, _ = potential.eval(values, gradient=True)
            ctx.save_for_backward(torch.from_numpy(np.array(gradient, copy=True)))
        else:
            value = potential.eval_value(values)
        return torch.tensor(value, dtype=torch.float64)

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        (gradient,) = ctx.saved_tensors
        return upstream * gradient, None, None


class _Response(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, model):
        ctx.values = _input(x)
        ctx.model = model
        value = model.eval_value(ctx.values)
        return torch.from_numpy(np.array(value, copy=True))

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        gradient = ctx.model.eval_vjp(ctx.values, _input(upstream))
        return torch.from_numpy(np.array(gradient, copy=True)), None

    @staticmethod
    def jvp(ctx, tangent, _model):
        product = ctx.model.eval_jvp(ctx.values, _input(tangent))
        return torch.from_numpy(np.array(product, copy=True))


def energy(potential, x):
    """Evaluate an Energy; differentiate with respect to x only."""
    if not isinstance(x, torch.Tensor):
        raise TypeError("expected a torch.Tensor")
    return _Energy.apply(x, potential, torch.is_grad_enabled() and x.requires_grad)


def response(model, x):
    """Evaluate a Response; reverse/forward AD use its VJP/JVP, differentiating x only."""
    if not isinstance(x, torch.Tensor):
        raise TypeError("expected a torch.Tensor")
    return _Response.apply(x, model)
