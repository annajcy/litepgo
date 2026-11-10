"""CPU float64 PyTorch adapters: Torch materials for native FEM and outer autograd.

TorchVolumetricMaterial and TorchShellMaterial turn a Torch energy density into a
litepgo.fem Python material; native FEM assembles it for every basis and quadrature.
The energy/response functions are first-order outer autograd adapters.

Only the explicit tensor input is differentiable. Callbacks and model configuration
must remain fixed from forward through differentiation. Each Response forward saves an independent evaluation with its input copy and
reusable state; backward and forward AD request its products. No response
Jacobian is saved.
"""

import numpy as np
import torch
from torch.autograd.function import once_differentiable

from litepgo import fem

__all__ = [
    "energy",
    "response",
    "TorchVolumetricMaterial",
    "TorchShellMaterial",
]


def _derivatives(material, inputs, active, mixed, outputs):
    """Evaluate material.W at NumPy inputs; return the requested entries as NumPy arrays.

    inputs: ordered {name: array}; active: the differentiated input; mixed: {output: input}
    for second derivatives. Missing dependencies give zero derivatives.
    """
    second = [name for name in outputs if mixed.get(name) is not None]
    need_first = any(name != "W" for name in outputs)
    # Native callbacks may run under no_grad/inference_mode set by the caller.
    with torch.inference_mode(False), torch.enable_grad():
        tensors = {
            name: torch.tensor(value, dtype=torch.float64).requires_grad_(
                (name == active and need_first) or any(mixed[o] == name for o in second)
            )
            for name, value in inputs.items()
        }
        W = material.W(*tensors.values())
        count = len(next(iter(inputs.values())))
        if not isinstance(W, torch.Tensor) or W.shape != (count,) or W.dtype != torch.float64:
            raise ValueError(f"{type(material).__name__}.W must return a float64 tensor of shape ({count},)")
        result = {}
        if "W" in outputs:
            result["W"] = W.detach().cpu().numpy()
        if not need_first:
            return result
        x = tensors[active]
        P = _grad(W.sum(), x, create_graph=bool(second))
        first = next(name for name, source in mixed.items() if source is None)
        result[first] = P.detach().cpu().numpy()
        rows = P.reshape(count, -1)
        for name in second:
            wrt = tensors[mixed[name]]
            columns = [_grad(rows[:, i].sum(), wrt, retain_graph=True) for i in range(rows.shape[1])]
            block = torch.stack([c.reshape(count, -1) for c in columns], dim=1)
            result[name] = block.reshape(*P.shape, *wrt.shape[1:]).detach().cpu().numpy()
        return result


def _grad(output, wrt, **options):
    """d output / d wrt, or zeros when output does not depend on wrt."""
    if not output.requires_grad:
        return torch.zeros_like(wrt)
    (gradient,) = torch.autograd.grad(output, wrt, allow_unused=True, **options)
    return torch.zeros_like(wrt) if gradient is None else gradient


class TorchVolumetricMaterial(fem.PyVolumetricMaterial):
    """A PyVolumetricMaterial whose density is written in PyTorch.

    Subclass it, set ``num_parameters`` and define ``W(self, F, Fp, theta)`` as an instance method taking CPU float64 tensors (N,3,3), (N,3,3), (N,k) and returning
    the (N,) density per original reference volume. Rows must be independent and W
    twice differentiable in the inputs whose derivatives are requested. Keep fixed
    state such as an nn.Module on the material instance. Concurrent evaluations
    must not mutate shared state.

    eval builds one graph per batch: dW/dF by one backward pass; d2W_dF2,
    d2W_dF_dtheta and d2W_dF_dFp by nine more. Network weights are constants here;
    use the outer energy/response adapters to differentiate a scalar loss.
    """

    def W(self, F, Fp, theta):
        raise NotImplementedError("TorchVolumetricMaterial subclasses must define W(F, Fp, theta)")

    def eval(self, F, Fp, theta, *, outputs):
        mixed = {"dW_dF": None, "d2W_dF2": "F", "d2W_dF_dtheta": "theta", "d2W_dF_dFp": "Fp"}
        return _derivatives(self, {"F": F, "Fp": Fp, "theta": theta}, "F", mixed, outputs)


class TorchShellMaterial(fem.PyShellMaterial):
    """A PyShellMaterial whose density is written in PyTorch.

    Subclass it, set ``num_parameters`` and define ``W(ab, ab_bar, thickness, theta)``
    taking CPU float64 tensors (N,6), (N,6), (N,), (N,k) with packed forms
    (a11,a12,a22,b11,b12,b22) and returning the (N,) density per original reference
    area. Autograd on the packed vector gives the independent, unscaled mixed-entry
    derivatives FEM expects. Second derivatives take six more backward passes.
    """

    def W(self, ab, ab_bar, thickness, theta):
        raise NotImplementedError("TorchShellMaterial subclasses must define W(ab, ab_bar, thickness, theta)")

    def eval(self, ab, ab_bar, thickness, theta, *, outputs):
        mixed = {"dW_dab": None, "d2W_dab2": "ab", "d2W_dab_dtheta": "theta", "d2W_dab_dabarbbar": "ab_bar"}
        inputs = {"ab": ab, "ab_bar": ab_bar, "thickness": thickness, "theta": theta}
        return _derivatives(self, inputs, "ab", mixed, outputs)


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
        evaluation = potential.evaluate(values)
        if need_gradient:
            gradient = evaluation.gradient()
            ctx.save_for_backward(torch.from_numpy(np.array(gradient, copy=True)))
        value = evaluation.value
        return torch.tensor(value, dtype=torch.float64)

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        (gradient,) = ctx.saved_tensors
        return upstream * gradient, None, None


class _Response(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, model):
        ctx.evaluation = model.evaluate(_input(x))
        value = ctx.evaluation.value
        return torch.from_numpy(np.array(value, copy=True))

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        gradient = ctx.evaluation.vjp(_input(upstream))
        return torch.from_numpy(np.array(gradient, copy=True)), None

    @staticmethod
    def jvp(ctx, tangent, _model):
        product = ctx.evaluation.jvp(_input(tangent))
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
