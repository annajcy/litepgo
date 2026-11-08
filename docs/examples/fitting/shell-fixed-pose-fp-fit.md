# Fixed-pose shell plastic stretches

[Back to Material and plastic fitting](./index.md)

**Experiment.** Use the same flat-sheet prescribed displacement as the shell
material case, fixed $(\lambda_{ps},\mu)=(2,3)$, and thickness 0.1. Fit

$$
F_p(p)=\operatorname{diag}(e^{p_0},e^{p_1},1),\qquad
s_{\rm true}=(1.06,0.95),\quad p_{\rm initial}=0.
$$

Targets come from `ShellFemEnergy` with fixed per-face plastic matrices,
keeping current positions and **original** area weights unchanged. The example
checks the fixed-pose force Response against that energy gradient before fitting. The residual is normalized by the target force norm.

**API walkthrough.** `ShellFemForceFpResponse[KoiterStVK]` exposes nine plastic
columns through a width-9 constant parameter map. Compose it with
`LogDiagonalFpResponse(2)` to obtain the two-column log-parameter Jacobian.
`Residual` only subtracts the target and divides by its norm.

**Expected result.** Recover `(1.06,0.95)`, with centered-difference Jacobian
agreement and rank checks at initial/truth/fitted states in the test suite. The flat rest stencil
cannot identify a normal plastic input direction, and left rotations preserve
natural forms; restricting to two known in-plane axes avoids those ambiguities.
No inner displacement solve, thickness fit, or plastic evolution occurs.

## Reading the example

Read `make_problem` for the mesh, prescribed pose, and synthetic target;
`Residual` (or `MaterialResidual`) for normalization; then `main` for fitting
and reporting. Numerical derivative and identifiability checks live in
`tests/python/test_fitting_example_derivatives.py`.

## Run and inspect

```sh
cmake --build build/release --target shell_fixed_pose_fp_fit litepgo_python
build/release/examples/fem/fitting/shell_fixed_pose_fp_fit
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/fitting/shell_fixed_pose_fp_fit.py
```

::: details Complete Python source — `shell_fixed_pose_fp_fit.py`

<<< ../../../examples/fem/fitting/shell_fixed_pose_fp_fit.py

:::

::: details Complete C++ source — `shell_fixed_pose_fp_fit.cpp`

<<< ../../../examples/fem/fitting/shell_fixed_pose_fp_fit.cpp

:::
