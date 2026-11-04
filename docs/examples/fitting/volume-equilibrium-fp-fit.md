# Equilibrium volume plastic stretches

[Back to Material and plastic fitting](./index.md)

**Experiment.** Use `assets/box-tet.veg`, fixed material `(2,3)`, bottom fixed,
and top displaced by $(0.04H,0,0.05H)$. Fit three shared positive diagonal
plastic stretches with truth `(1.04,0.97,1.02)`, starting from identity in log
coordinates. Observe both free displacements and fixed-DOF reactions at equilibrium.
Each observation block is divided by its fixed target norm, floored at $10^{-8}$.

**Read the example in this order.** C++ and Python use the same structure, with
all helpers in the example file:

1. `make_problem()` prepares geometry, fixed material fields, parameter maps, and
   boundary displacements. `Problem` only stores these inputs.
2. `make_factories(problem)` creates a shared `ElementwiseResponse` using
   `LogDiagonalFpResponse`, then defines `make_energy(p)` and `make_response(u)`
   together. The former builds Fp fields; the latter composes the native force
   response with that same parameter transform.
3. `measure(...)` solves equilibrium, computes force and its total derivative,
   then concatenates free displacement and constrained reactions.
4. `Residual` subtracts the target and divides by fixed observation scales.
5. `main()` generates the target, fits from zero log stretches, and predicts a
   second load. It checks solver convergence and prints the result.

The key observation derivative stays visible in `measure`:

```python
reaction_jacobian = force_parameter_jacobian + stiffness @ state_jacobian
```

The first term is the force derivative at fixed displacement. The second accounts
for the displacement change along the equilibrium branch. Only fixed rows of this
sum are observed as reactions; free rows of `state_jacobian` supply displacement
observations. See [parameter transforms](../../guide/energy/parameter-transforms.md)
for the shared parameterization interface.

**Expected result.** Recover `(1.04, 0.97, 1.02)` and predict observations at load
factor `0.6`. Output includes `truth`, `fitted`, `converged`, `parameter_error`,
`relative_residual`, and `held_out_error`. The model fixes plastic principal axes
and does not assume a full nine-entry matrix is identifiable.

Finite-difference checks, rank checks at initial/truth/fitted parameters, nonzero
state sensitivity, prescribed-value checks, and fitting-error thresholds live in
`tests/python/test_equilibrium_fp_fit_examples.py`. These tests exercise the
Python example functions directly and compare the complete C++ and Python fitting
outputs. The example no longer prints `state_fd_error`, `reaction_fd_error`, or
rank ratios. The physical experiment, solver settings, and fitting metrics are
unchanged.

## Run and inspect

```sh
cmake --build build/release --target volume_equilibrium_fp_fit litepgo_python
build/release/examples/fem/fitting/volume_equilibrium_fp_fit
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/fitting/volume_equilibrium_fp_fit.py
```

::: details Complete Python source — `volume_equilibrium_fp_fit.py`

<<< ../../../examples/fem/fitting/volume_equilibrium_fp_fit.py

:::

::: details Complete C++ source — `volume_equilibrium_fp_fit.cpp`

<<< ../../../examples/fem/fitting/volume_equilibrium_fp_fit.cpp

:::

## Run the regression checks

```sh
uv run --no-sync python -m pytest tests/python/test_equilibrium_fp_fit_examples.py -k volume
```
