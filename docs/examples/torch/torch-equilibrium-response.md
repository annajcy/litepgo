# Torch force responses through equilibrium

[Back to PyTorch examples](./index.md)

This example compares Torch and native equilibrium derivatives for both physical
material parameters and a full column-major plastic matrix. It reuses the
[Neo-Hookean density and two-tet setup](./torch-neohookean.md) from the adjacent
`torch_neohookean.py`; keep both files together.

Read `make_equilibrium` for the setup and the two factories:

- `make_energy(p)` builds matching material/Fp fields and a scalar energy.
- `make_response(u)` holds displacement fixed and returns the corresponding
  `TorchVolumetricFemMaterialResponse` or `TorchVolumetricFemFpResponse`.
- `StaticEquilibriumResponse` uses those objects to compute state values,
  explicit Jacobians, directional JVPs, and adjoint VJPs.

The material experiment shares two physical inputs. The plastic experiment
shares nine physical inputs, including off-diagonal entries; the remaining
material fields stay fixed. Both use the same constrained two-tetrahedron
problem and report state/Jacobian/JVP/VJP differences against native FEM.
This demonstrates sensitivity, not identifiability or fitting all nine plastic
components from the three free displacement DOFs.

Force-response products directly contract mixed derivatives of W in Torch,
without assembling parameter Jacobians. The equilibrium products then solve
one directional or adjoint stiffness system. Tests disable the explicit
force-Jacobian path to verify this behavior, compare re-solved finite differences,
and exercise the outer Torch bridge and composed log parameterizations.

The example supports TetP1/centroid and CPU float64. Network weights are fixed;
`MaterialResponse` differentiates physical theta, not implicitly captured weights.
See [the response API](../../guide/torch.md#torch-material-and-plastic-force-responses).

## Run and inspect

```sh
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/fitting/torch_equilibrium_response.py
uv run --no-sync python -m pytest -q tests/python/test_torch_fem_response.py
```

::: details Complete Python source — `torch_equilibrium_response.py`

<<< ../../../examples/fem/fitting/torch_equilibrium_response.py

:::
