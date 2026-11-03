# Fixed-pose shell material

[Back to Material and plastic fitting](./index.md)

**Experiment.** Read `assets/shell.obj`, hold thickness at 0.1, and prescribe a
stretch/shear/saddle pose. For centered coordinates $(X,Y)$ and bounding
size $s$,

$$
u_0(X,Y)=(0.08X+0.04Y,\ -0.03Y,\ 0.15XY/s).
$$

Fit the shared plane-stress parameters $(\lambda_{ps},\mu)$. Targets are all
nodal force components from the energy gradient at $(2,3)$. The residual is
$(f(\theta)-f_{\rm obs})/\|f_{\rm obs}\|$, starting at $(0.7,1.2)$.
These parameters are optimized directly, without a log transform.

**API walkthrough.** Build stencils, `TriangleMidEdgeRestData`, and a
`ShellFemEnergy[KoiterStVK]` to generate targets. A
`ShellFemForceMaterialResponse[KoiterStVK]` with `ConstantDofMap(topology,2)`
provides predictions/Jacobians. A `FixedPatternResponse` wrapper subtracts and
scales targets, rejecting invalid material trials, then passes to Gauss–Newton.

**Expected result.** Both parameters recover `(2,3)`. The program prints relative
parameter/force errors and $\sigma_{\min}/\sigma_{\max}$, requiring a ratio
above $10^{-6}$. At fixed pose Koiter force is linear in these parameters;
there is no inner equilibrium solve or unknown thickness.

## Run and inspect

```sh
cmake --build build/release --target shell_fixed_pose_material_fit litepgo_python
build/release/examples/fem/fitting/shell_fixed_pose_material_fit
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/fitting/shell_fixed_pose_material_fit.py
```

::: details Complete Python source — `shell_fixed_pose_material_fit.py`

<<< ../../../examples/fem/fitting/shell_fixed_pose_material_fit.py

:::

::: details Complete C++ source — `shell_fixed_pose_material_fit.cpp`

<<< ../../../examples/fem/fitting/shell_fixed_pose_material_fit.cpp

:::
