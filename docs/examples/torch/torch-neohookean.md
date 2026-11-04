# A PyTorch constitutive energy

[Back to PyTorch examples](./index.md)

Write `material(F, Fp, theta)` in PyTorch, then use
`litepgo.torch.TorchVolumetricFemEnergy` as an ordinary native `Energy`.
This example compares the same Neo-Hookean density with the native implementation;
Torch supplies material derivatives while C++ assembles and solves the FEM system.

The model has two tetrahedra sharing a face, distinct material parameters, and
non-diagonal positive-determinant plastic matrices. This exercises matrix packing,
reference-volume weights, and shared-node assembly. Its density is

$$
F_e=FF_p^{-1},\quad
W=\det(F_p)\left[\frac\mu2(\|F_e\|_F^2-3)-\mu\log\det(F_e)
+\frac\lambda2(\log\det(F_e))^2\right].
$$

`Fp` fields store column-major vectors; the callable receives ordinary `(B,3,3)`
matrices. The callable rejects lambda < 0 or mu <= 0, matching the native material domain.
It includes `det(Fp)` itself; the adapter only multiplies the reference
integration weight. Field lists are sampled and snapshotted at the centroid.
The material object is retained and must remain fixed during evaluation/solving.

Read `NeoHookean.forward` for the formula, `make_problem` for the shared native/Torch
setup, and `main` for derivative comparisons and constrained equilibrium. The
example reports absolute energy, force, stiffness, and equilibrium differences.
Tests enforce tolerances and additionally cover finite differences, compressed
states with Hessian projection, inversion-safe steps, batches, invalid inputs,
and a small neural density with fixed weights.

This first version supports TetP1, centroid quadrature, and CPU float64. It computes
local second derivatives in Torch; the separate `litepgo.torch.energy` bridge still
supports only first-order outer differentiation. Network training through equilibrium
requires an explicit weight mapping and is not supplied by this energy alone.
For physical theta and Fp optimization, see the
[force-response equilibrium example](./torch-equilibrium-response.md).
See [the constitutive API](../../guide/torch.md#constitutive-energy-written-in-torch).

## Run and inspect

```sh
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/fitting/torch_neohookean.py
uv run --no-sync python -m pytest -q tests/python/test_torch_fem.py
```

::: details Complete Python source — `torch_neohookean.py`

<<< ../../../examples/fem/fitting/torch_neohookean.py

:::
