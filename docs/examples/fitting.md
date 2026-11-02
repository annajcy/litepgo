# Material and plastic-state identification

These seven programs fit synthetic noiseless observations. They print recovery,
residual, and validation diagnostics and do not write meshes. A zero exit means
the program's convergence/recovery checks passed. Read the formulation below
before selecting an experiment: fixed-pose force fitting and equilibrium fitting
use different derivatives.

## Shared formulation

Let $p$ be the chosen parameter vector, $E(u,p)$ elastic energy, and
$g=\nabla_uE$ the positive-gradient force. All cases minimize

$$
\Phi(p)=\tfrac12\|r(p)\|^2,\qquad r=S^{-1}(y(p)-y_{\rm obs}),
$$

with fixed training scales $S$. At a prescribed pose, $y=g(u_0,p)$ and
$J_y=B=\partial g/\partial p|_{u_0}$. At equilibrium, let $u_f$ be free
DOFs and $C$ select fixed-DOF reactions. Then

$$
g_f(u^*,p)=0,\qquad K_{ff}U_f=-B_f,\quad U_c=0,
$$

$$
\frac{d(Cg)}{dp}=C(B+KU),\quad K=\nabla_u^2E.
$$

`StaticEquilibriumResponse` computes $u^*,U$. Force-response objects supply
$B$; they do not include the state change by themselves. The exact unshifted
stiffness $K$ is used for sensitivities. Fixed indices and values are
parameter-independent.

Volumetric cases use the [Neo-Hookean solid energy](../guide/fem/volume.md),
including $W=\det F_p\,\psi(FF_p^{-1})$. Shell cases use the
[Koiter natural-form energy](../guide/fem/shell.md), with original area and
thickness fixed. Parameterized positive stretches $s_j=e^{p_j}$ multiply
selected derivative columns by $s_j$. `vec(Fp)` uses column-major order, so
diagonal entries occupy columns 0, 4, 8.

These are locally checked recovery experiments, not evidence that arbitrary
materials or full plastic matrices are identifiable. No noise model, spatial
regularization, or plastic evolution is added.

All commands run from the repository root after [configuring the release build](../guide/build.md). Python commands explicitly select that build. Expand the source sections for complete programs, including imports, setup, checks, and output handling.

## Fixed-pose shell material

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

### Run and inspect

```sh
cmake --build build/release --target shell_fixed_pose_material_fit litepgo_dev
build/release/examples/fem/fitting/shell_fixed_pose_material_fit
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/fitting/shell_fixed_pose_material_fit.py
```

::: details Complete Python source — `shell_fixed_pose_material_fit.py`

<<< ../../examples/fem/fitting/shell_fixed_pose_material_fit.py

:::

::: details Complete C++ source — `shell_fixed_pose_material_fit.cpp`

<<< ../../examples/fem/fitting/shell_fixed_pose_material_fit.cpp

:::


## Fixed-pose volume plastic stretches

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
plastic matrix. The wrapper restricts and differentiates the log parameterization:

$$
J_p=J_{\operatorname{vec}F_p}[:,(0,4,8)]\operatorname{diag}(e^p)/\|f_{\rm obs}\|.
$$

Return `None` for unrepresentable exponential trials. Gauss–Newton receives the
three-column residual object, not the unconstrained nine-column force response.

**Expected result.** The three stretches are recovered. Checks compare the
composed Jacobian to centered finite differences and require local full column
rank at initial, true, and fitted states. Output reports these diagnostics and
relative force/parameter errors. There is no equilibrium or yield-law update;
principal plastic axes are assumed known.

### Run and inspect

```sh
cmake --build build/release --target volume_fixed_pose_fp_fit litepgo_dev
build/release/examples/fem/fitting/volume_fixed_pose_fp_fit
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/fitting/volume_fixed_pose_fp_fit.py
```

::: details Complete Python source — `volume_fixed_pose_fp_fit.py`

<<< ../../examples/fem/fitting/volume_fixed_pose_fp_fit.py

:::

::: details Complete C++ source — `volume_fixed_pose_fp_fit.cpp`

<<< ../../examples/fem/fitting/volume_fixed_pose_fp_fit.cpp

:::


## Fixed-pose shell plastic stretches

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

### Run and inspect

```sh
cmake --build build/release --target shell_fixed_pose_fp_fit litepgo_dev
build/release/examples/fem/fitting/shell_fixed_pose_fp_fit
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/fitting/shell_fixed_pose_fp_fit.py
```

::: details Complete Python source — `shell_fixed_pose_fp_fit.py`

<<< ../../examples/fem/fitting/shell_fixed_pose_fp_fit.py

:::

::: details Complete C++ source — `shell_fixed_pose_fp_fit.cpp`

<<< ../../examples/fem/fitting/shell_fixed_pose_fp_fit.cpp

:::


## Equilibrium volume material

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

### Run and inspect

```sh
cmake --build build/release --target volume_equilibrium_material_fit litepgo_dev
build/release/examples/fem/fitting/volume_equilibrium_material_fit
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/fitting/volume_equilibrium_material_fit.py
```

::: details Complete Python source — `volume_equilibrium_material_fit.py`

<<< ../../examples/fem/fitting/volume_equilibrium_material_fit.py

:::

::: details Complete C++ source — `volume_equilibrium_material_fit.cpp`

<<< ../../examples/fem/fitting/volume_equilibrium_material_fit.cpp

:::


## Equilibrium shell material

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
`ShellFemForceMaterialResponse` partial feed `StaticEquilibriumResponse`.
Select fixed rows of $B+KU$ for the reaction Jacobian, then normalize and
pass to Gauss–Newton. Nonuniform boundary loading excites material-dependent
center motion, making the state-sensitivity term observable.

**Expected result.** Recover `(2,3)` and predict reactions at 0.6 times the
training boundary load. Checks include nonzero state sensitivity, local rank,
exact boundary preservation, free equilibrium, and independently re-solved
finite differences of displacement and reactions. No geometry or thickness is
optimized.

### Run and inspect

```sh
cmake --build build/release --target shell_equilibrium_material_fit litepgo_dev
build/release/examples/fem/fitting/shell_equilibrium_material_fit
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/fitting/shell_equilibrium_material_fit.py
```

::: details Complete Python source — `shell_equilibrium_material_fit.py`

<<< ../../examples/fem/fitting/shell_equilibrium_material_fit.py

:::

::: details Complete C++ source — `shell_equilibrium_material_fit.cpp`

<<< ../../examples/fem/fitting/shell_equilibrium_material_fit.cpp

:::


## Equilibrium volume plastic stretches

**Experiment.** Use `assets/box-tet.veg`, fixed material `(2,3)`, bottom fixed,
and top displaced by $(0.04H,0,0.05H)$. Fit three shared positive diagonal
plastic stretches with truth `(1.04,0.97,1.02)`, starting from identity in log
coordinates. Observe both free displacements and fixed-DOF reactions at equilibrium.
Each observation block is divided by its fixed target norm, floored at $10^{-8}$.

**API walkthrough.** `VolumetricFemForceFpResponse` supplies the nine-column
fixed-pose derivative. Select columns 0/4/8 and apply $e^p$ to obtain $B_p$.
`StaticEquilibriumResponse(3,...)` solves $K_{ff}U_f=-B_{p,f}$.
Stack free rows of $U$ and fixed rows of $B_p+KU$ to form the total
observation Jacobian for the outer `FixedPatternResponse`.

**Expected result.** Recover all three stretches and predict displacement and
reaction observations at load factor 0.6. The program checks full column rank
at initial/truth/fitted states, nonzero state sensitivity, fixed values, and
re-solved finite differences of both observation blocks. The model fixes
plastic principal axes and does not assume a full nine-entry matrix is identifiable.

### Run and inspect

```sh
cmake --build build/release --target volume_equilibrium_fp_fit litepgo_dev
build/release/examples/fem/fitting/volume_equilibrium_fp_fit
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/fitting/volume_equilibrium_fp_fit.py
```

::: details Complete Python source — `volume_equilibrium_fp_fit.py`

<<< ../../examples/fem/fitting/volume_equilibrium_fp_fit.py

:::

::: details Complete C++ source — `volume_equilibrium_fp_fit.cpp`

<<< ../../examples/fem/fitting/volume_equilibrium_fp_fit.cpp

:::


## Equilibrium shell plastic stretches

**Experiment.** Use the generated 3-by-3 shell patch and boundary field from the
shell material experiment, thickness 0.1, and fixed material `(2,3)`. Fit
$F_p=\operatorname{diag}(e^{p_0},e^{p_1},1)$ with truth `(1.06,0.95)`, starting
at identity. Observe the free center displacement and 24 boundary reactions,
normalized by separate fixed target norms.

**API walkthrough.** Rebuild `ShellFemEnergy` with the per-face natural-state
matrices for each trial. `ShellFemForceFpResponse` plus columns 0/4 and exponential
factors supplies $B_p$. `StaticEquilibriumResponse(2,...)` supplies $U$.
The combined Jacobian stacks $U_f$ and $(B_p+KU)_c$, not merely the
fixed-pose plastic-force partial.

**Expected result.** Recover `(1.06,0.95)` and predict both observation blocks
at load factor 0.6. Re-solved finite differences, rank, fixed-value, equilibrium,
and recovery checks gate success. Tight inner tolerances are used because
normalizing a small displacement observation amplifies derivative error.
Original shell areas/thickness remain fixed and no normal plastic stretch or
plastic evolution is fitted.

### Run and inspect

```sh
cmake --build build/release --target shell_equilibrium_fp_fit litepgo_dev
build/release/examples/fem/fitting/shell_equilibrium_fp_fit
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/fitting/shell_equilibrium_fp_fit.py
```

::: details Complete Python source — `shell_equilibrium_fp_fit.py`

<<< ../../examples/fem/fitting/shell_equilibrium_fp_fit.py

:::

::: details Complete C++ source — `shell_equilibrium_fp_fit.cpp`

<<< ../../examples/fem/fitting/shell_equilibrium_fp_fit.cpp

:::
