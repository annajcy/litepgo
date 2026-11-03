"""CPU float64, first-order PyTorch adapters for Energy and Response.

Only the explicit tensor input is differentiable. Callbacks and model configuration
must remain fixed during evaluation. Sparse Jacobians are saved per forward call.
"""

import numpy as np
import torch
from torch.autograd.function import once_differentiable

__all__ = ["energy", "response"]


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
    def forward(ctx, x, model, need_gradient):
        values = _input(x)
        if need_gradient:
            model.update_pattern(values)
            value, jacobian = model.eval(values, need_jacobian=True)
            ctx.jacobian = jacobian.copy()
        else:
            value = model.eval_value(values)
        return torch.from_numpy(np.array(value, copy=True))

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        gradient = ctx.jacobian.T @ upstream.detach().numpy()
        return torch.from_numpy(np.array(gradient, copy=True)), None, None


def energy(potential, x):
    """Evaluate an Energy; differentiate with respect to x only."""
    if not isinstance(x, torch.Tensor):
        raise TypeError("expected a torch.Tensor")
    return _Energy.apply(x, potential, torch.is_grad_enabled() and x.requires_grad)


def response(model, x):
    """Evaluate a Response, including static equilibrium; differentiate x only."""
    if not isinstance(x, torch.Tensor):
        raise TypeError("expected a torch.Tensor")
    return _Response.apply(x, model, torch.is_grad_enabled() and x.requires_grad)
