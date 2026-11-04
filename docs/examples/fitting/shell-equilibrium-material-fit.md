# Equilibrium shell material

[Back to Material and plastic fitting](./index.md)

**Experiment.** Generate a 3-by-3 vertex patch with eight triangles and thickness
0.1. Prescribe boundary displacement

$$
u_D(X,Y)=(0.08X+0.025X^2+0.04Y,\ 0.03Y+0.02XY,\ 0.08XY+0.03X^2).
$$

The center vertex's three DOFs remain free. Observe the 24 boundary reactions at
$(\lambda_{ps},\mu)=(2,3)$; divide by their target Euclidean norm. Optimize
parameters directly from `(0.7,1.2)`, rejecting $\mu\le0$ or
$\lambda_{ps}+\mu\le0$.

**API walkthrough.** The `ShellFemEnergy` factory and
`ShellFemForceMaterialResponse` factory feed `StaticEquilibriumResponse`.
Select fixed rows of $B+KU$ for the reaction Jacobian, then normalize and
pass to Gauss–Newton. Nonuniform boundary loading excites material-dependent
center motion, making the state-sensitivity term observable.

**Expected result.** Recover `(2,3)` and predict reactions at 0.6 times the
training boundary load. Tests cover nonzero state sensitivity, local rank,
exact boundary preservation, free equilibrium, and independently re-solved
finite differences of displacement and reactions. No geometry or thickness is
optimized.

## Reading the example

Read `make_problem` for geometry and constraints, `make_factories` for the
physical model, `measure` for equilibrium and reactions, and `Residual` for
normalization. `main` generates observations, fits, and predicts a held-out load.
Derivative, rank, and boundary checks live in the fitting tests.

## Run and inspect

```sh
cmake --build build/release --target shell_equilibrium_material_fit litepgo_python
build/release/examples/fem/fitting/shell_equilibrium_material_fit
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/fitting/shell_equilibrium_material_fit.py
```

::: details Complete Python source — `shell_equilibrium_material_fit.py`

<<< ../../../examples/fem/fitting/shell_equilibrium_material_fit.py

:::

::: details Complete C++ source — `shell_equilibrium_material_fit.cpp`

<<< ../../../examples/fem/fitting/shell_equilibrium_material_fit.cpp

:::
