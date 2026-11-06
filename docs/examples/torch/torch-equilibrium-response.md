# Torch force responses through equilibrium

[Back to PyTorch examples](./index.md)

This example compares Torch and native equilibrium derivatives for both physical
material parameters and a full column-major plastic matrix. It reuses the
[Neo-Hookean material and two-tet setup](./torch-neohookean.md) from the adjacent
`torch_neohookean.py`; keep both files together.

Read `make_equilibrium` for the setup and the two factories. One `material`
variable selects `TorchNeoHookean` or the native `NeoHookean`; everything else
is the same code:

- `make_energy(p)` builds matching material/Fp fields and a scalar energy.
- `make_response(u)` holds displacement fixed and returns the corresponding
  `VolumetricFemForceMaterialResponse` or `VolumetricFemForceFpResponse`.
- `StaticEquilibriumResponse` uses those objects to compute state values,
  explicit Jacobians, directional JVPs, and adjoint VJPs.

The responses ask the Torch material for $\partial^2W/\partial F\partial\theta$
or $\partial^2W/\partial F\partial F_p$ in one batched call and assemble explicit
parameter Jacobians natively. The material experiment shares two physical
inputs. The plastic experiment shares nine physical inputs, including
off-diagonal entries; the remaining material fields stay fixed. Both use the
same constrained two-tetrahedron problem and report state/Jacobian/JVP/VJP
differences against native FEM. This demonstrates sensitivity, not
identifiability or fitting all nine plastic components from the three free
displacement DOFs.

Network weights are fixed; `VolumetricFemForceMaterialResponse` differentiates
physical theta, not captured weights.
See [the response API](../../guide/torch.md#force-responses-of-a-torch-material).

## Run and inspect

```sh
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/fitting/torch_equilibrium_response.py
PYTHONPATH=build/release/python uv run --no-sync python -m pytest -q tests/python/test_py_material_examples.py
```

::: details Complete Python source — `torch_equilibrium_response.py`

<<< ../../../examples/fem/fitting/torch_equilibrium_response.py

:::
