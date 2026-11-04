# Fixed-pose volume plastic stretches

[Back to Material and plastic fitting](./index.md)

**Experiment.** Read `assets/box-tet.veg`, hold Neo-Hookean $(\lambda,\mu)=(2,3)$,
and prescribe $u_0(X)=(F-I)(X-c)$, with $c$ the nodal mean and

$$
F=\begin{bmatrix}1.12&0.08&0.02\\0.01&0.96&0.06\\0.03&0&1.07\end{bmatrix},\qquad
F_p(p)=\operatorname{diag}(e^{p_0},e^{p_1},e^{p_2}).
$$

Truth stretches are `(1.08,0.94,1.03)`. Start at $p=0$, or identity plastic
state. Observe all nodal forces and divide the force residual by target norm.

**API walkthrough.** Targets come from `VolumetricFemEnergy.eval_gradient`.
`VolumetricFemForceFpResponse` uses `ConstantDofMap(topology,9)` for the full
plastic matrix. `ComposedResponse(force, LogDiagonalFpResponse(3))` restricts
and differentiates the log parameterization:

$$
J_p=J_{\operatorname{vec}F_p}[:,(0,4,8)]\operatorname{diag}(e^p)/\|f_{\rm obs}\|.
$$

`LogDiagonalFpResponse` rejects unrepresentable exponential trials. Gauss–Newton receives the
three-column residual object, not the unconstrained nine-column force response.

**Expected result.** The three stretches are recovered. Checks compare the
composed Jacobian to centered finite differences and require local full column
rank at initial, true, and fitted states in the test suite. Output reports
relative force/parameter errors. There is no equilibrium or yield-law update;
principal plastic axes are assumed known.

## Reading the example

Read `make_problem` for the mesh, prescribed pose, and synthetic target;
`Residual` (or `MaterialResidual`) for normalization; then `main` for fitting
and reporting. Numerical derivative and identifiability checks live in
`tests/python/test_fitting_example_derivatives.py`.

## Run and inspect

```sh
cmake --build build/release --target volume_fixed_pose_fp_fit litepgo_python
build/release/examples/fem/fitting/volume_fixed_pose_fp_fit
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/fitting/volume_fixed_pose_fp_fit.py
```

::: details Complete Python source — `volume_fixed_pose_fp_fit.py`

<<< ../../../examples/fem/fitting/volume_fixed_pose_fp_fit.py

:::

::: details Complete C++ source — `volume_fixed_pose_fp_fit.cpp`

<<< ../../../examples/fem/fitting/volume_fixed_pose_fp_fit.cpp

:::
