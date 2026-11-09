# A PyTorch constitutive energy

[Back to PyTorch examples](./index.md)

Write the density $W(F,F_p,\theta)$ in PyTorch as a `TorchVolumetricMaterial`
subclass and pass an instance as the first argument to `fem.VolumetricFemEnergy`.
This example compares the same Neo-Hookean density with the native implementation;
Torch supplies material derivatives while C++ computes deformation gradients,
integrates, assembles and solves.

The first model has two tetrahedra sharing a face, distinct material parameters,
and non-diagonal positive-determinant plastic matrices. This exercises matrix
packing, reference-volume weights, and shared-node assembly. The second uses the
bundled hexahedral box with Q1 displacement and 2×2×2 Gauss points: the same
material instance works with every compiled basis and quadrature. The density is

$$
F_e=FF_p^{-1},\quad
W=\det(F_p)\left[\frac\mu2(\|F_e\|_F^2-3)-\mu\log\det(F_e)
+\frac\lambda2(\log\det(F_e))^2\right].
$$

`Fp` fields store column-major vectors; `W` receives ordinary $(N,3,3)$
matrices, one row per quadrature point of a batch. It returns NaN for
$\lambda<0$ or $\mu\le0$, matching the native material domain: a non-finite
density rejects a Newton trial. It includes $\det F_p$ itself; native assembly
only multiplies the reference integration weight.

Read `TorchNeoHookean.W` for the formula, `make_problem` and `hex_box` for the
shared native/Torch setups, and `main` for derivative comparisons and constrained
equilibrium. The example reports absolute energy, force, stiffness, and
equilibrium differences. Tests enforce tolerances and additionally cover every
basis and quadrature, batches, the requested outputs, domain rejection, and
invalid materials. The contract is described in
[materials written in Python](../../guide/fem/python-materials.md).

For physical theta and Fp optimization, see the
[force-response equilibrium example](./torch-equilibrium-response.md).
Network weights captured by a material are constants; training them through
equilibrium requires an explicit weight mapping.

## Run and inspect

```sh
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/fitting/torch_neohookean.py
PYTHONPATH=build/release/python uv run --no-sync python -m pytest -q tests/python/test_py_material.py
```

::: details Complete Python source — `torch_neohookean.py`

<<< ../../../examples/fem/fitting/torch_neohookean.py

:::
