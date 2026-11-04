# Equilibrium volume material

[Back to Material and plastic fitting](./index.md)

**Experiment.** Read the tetrahedral bar, fix the bottom, and prescribe top-face
motion while side/interior DOFs relax. Two training experiments impose top
x-shear $0.08H$ and z-compression $-0.04H$, where $H$ is rest height.
The measured scalar for each is the sum of top-face gradient components in its
loading direction. Truth is $(\lambda,\mu)=(2,3)$, $F_p=I$.

Optimize $q$ with $\theta=e^q$, starting at $q=0$. Each of the two
reaction residuals is divided by its fixed observed magnitude, floored at
$10^{-8}$. Its derivative includes both $B+KU$ and the exponential factors.

**API walkthrough.** `make_energy(theta)` reconstructs material snapshots.
`VolumetricFemForceMaterialResponse` provides $B$ at each solved state.
`StaticEquilibriumResponse(2,...)` performs the inner displacement solve and
analytic sensitivity. The outer `FixedPatternResponse` selects/sums loaded
reaction rows, normalizes them, and calls Gauss–Newton.

**Expected result.** Recover `(2,3)` and predict a held-out compression of
$-0.07H$. The program checks fixed values, equilibrium convergence, and the
material scaling identity $(dR/d\theta)\theta=R$, then prints parameter,
residual, prediction errors, and forward-solve count. The observation/Jacobian
has only two parameters; it does not identify general spatial material fields.

## Run and inspect

```sh
cmake --build build/release --target volume_equilibrium_material_fit litepgo_python
build/release/examples/fem/fitting/volume_equilibrium_material_fit
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/fitting/volume_equilibrium_material_fit.py
```

::: details Complete Python source — `volume_equilibrium_material_fit.py`

<<< ../../../examples/fem/fitting/volume_equilibrium_material_fit.py

:::

::: details Complete C++ source — `volume_equilibrium_material_fit.cpp`

<<< ../../../examples/fem/fitting/volume_equilibrium_material_fit.cpp

:::
