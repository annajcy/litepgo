# Fixed-pose shell plastic stretches

[Back to Material and plastic fitting](./index.md)

**Experiment.** Use the same flat-sheet prescribed displacement as the shell
material case, fixed $(\lambda_{ps},\mu)=(2,3)$, and thickness 0.1. Fit

$$
F_p(p)=\operatorname{diag}(e^{p_0},e^{p_1},1),\qquad
s_{\rm true}=(1.06,0.95),\quad p_{\rm initial}=0.
$$

Targets are independently assembled from local shell energy gradients at the
transformed natural stencils, keeping current positions and **original** area
weights unchanged. The residual is normalized by the target force norm.

**API walkthrough.** `ShellFemForceFpResponse[KoiterStVK]` exposes nine plastic
columns through a width-9 constant parameter map. Select columns 0 and 4 and
multiply by the positive stretches for the log-parameter chain rule. Wrap this
in `FixedPatternResponse` for Gauss–Newton.

**Expected result.** Recover `(1.06,0.95)`, with centered-difference Jacobian
agreement and rank checks at initial/truth/fitted states. The flat rest stencil
cannot identify a normal plastic input direction, and left rotations preserve
natural forms; restricting to two known in-plane axes avoids those ambiguities.
No inner displacement solve, thickness fit, or plastic evolution occurs.

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
