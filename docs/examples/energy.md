# Scalar energies and vector responses

These four programs explain energy and response APIs before an equilibrium solve.
They generate their own data and write no meshes. A zero exit means their numerical
checks passed. The [energy/response guide](../guide/energy/index.md) defines
sparse patterns, callback shapes, and ownership.

All commands run from the repository root after [configuring the release build](../guide/build.md). Python commands explicitly select that build. Expand the source sections for complete programs, including imports, setup, checks, and output handling.

## Custom scalar energies

**Problem.** For $x\in\mathbb R^2$, evaluate

$$
E_0(x)=\tfrac12\|x-(1,2)\|^2,\qquad
E_1(x)=\tfrac12\|x\|^2+\tfrac14[\max(0,x_0+x_1)]^4.
$$

With $s=\max(0,x_0+x_1)$ and $\mathbf1=(1,1)^T$,

$$
\nabla E_0=x-(1,2),\quad H_0=I,\qquad
\nabla E_1=x+s^3\mathbf1,\quad H_1=I+3s^2\mathbf1\mathbf1^T.
$$

**API walkthrough.** `Quadratic` derives from `FixedPatternEnergy` with diagonal
structure. `Coupled` derives from `DynamicPatternEnergy`; `pattern_at` returns
full structure when $x_0+x_1>0$, diagonal otherwise. Both implement
`eval_values`, returning requested derivatives and `None` for unused outputs.
Both energies are convex, so Exact and Projected Hessians can coincide here.

Refresh structure with `update_pattern` before Hessian assembly. Value-only
calls do not need a refresh. C++ receives output pointers; Python returns owning
arrays. A second refresh at the same state reports no structural change.

**Expected result.** At $(1,1)$, $E_0=0.5$, $E_1=5$, and
$\nabla E_1=(9,9)$. The stored Hessian count follows `2 -> 4 -> 2` as the
active branch changes. Printed values and assertions check both derivatives
and structure.

### Run and inspect

```sh
cmake --build build/release --target custom_energy litepgo_dev
build/release/examples/energy/scalar/custom_energy
PYTHONPATH=build/release/python uv run --no-sync python examples/energy/scalar/custom_energy.py
```

::: details Complete Python source — `custom_energy.py`

<<< ../../examples/energy/scalar/custom_energy.py

:::

::: details Complete C++ source — `custom_energy.cpp`

<<< ../../examples/energy/scalar/custom_energy.cpp

:::


## Energy composition

**Problem.** Combine a regularizer and target springs, then change coordinates:

$$
F(x)=\tfrac14\|x\|^2+\|x-t\|^2,\quad t=(1,2,3)^T,\quad x=Tz+c,
$$

$$
T=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix},\quad
c=(0,0,1)^T,\quad z=(0.2,-0.3)^T,
$$

$$
\nabla_zF=T^T(2.5x-2t),\qquad H_z=2.5T^TT.
$$

**API walkthrough.** `QuadraticEnergy(I)` and `VertexPullingEnergy` supply the
terms. `WeightedEnergySum` weights them by 0.5 and 2.0. `AffineMappedEnergy(total,
T, c)` performs the pullback; refresh its pattern before requesting the Hessian.

**Expected result.** The program prints energy, a two-entry gradient, and a
2-by-2 Hessian, comparing each to the equations above. No optimization occurs;
this isolates composition and the affine chain rule.

### Run and inspect

```sh
cmake --build build/release --target energy_composition litepgo_dev
build/release/examples/energy/scalar/energy_composition
PYTHONPATH=build/release/python uv run --no-sync python examples/energy/scalar/energy_composition.py
```

::: details Complete Python source — `energy_composition.py`

<<< ../../examples/energy/scalar/energy_composition.py

:::

::: details Complete C++ source — `energy_composition.cpp`

<<< ../../examples/energy/scalar/energy_composition.cpp

:::


## Custom vector responses

**Problem.** Evaluate and solve linear and branch-dependent residuals:

$$
r_0(x)=\begin{bmatrix}x_0-1\\x_1-2\\x_0+x_1-3\end{bmatrix},\quad
J_0=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix},
$$

$$
r_1(x)=\begin{bmatrix}x_0-1\\x_1+\max(0,x_0)^2-2\end{bmatrix},\quad
J_1=\begin{bmatrix}1&0\\2\max(0,x_0)&1\end{bmatrix}.
$$

**API walkthrough.** Derive the linear class from `FixedPatternResponse` and the
nonlinear class from `DynamicPatternResponse`. Implement `eval_values` to return
`r` or `(r, J)` according to `need_jacobian`. The dynamic class adds/removes the
lower-left entry. Pass the residual object to `gauss_newton_solve`; its pattern
is refreshed automatically at current iterates.

**Expected result.** The linear residual vanishes at $(1,2)$, with four
Jacobian entries. The nonlinear solve goes from $(-1,-1)$ to $(1,1)$,
with three active Jacobian entries. Assertions check dimensions, structural
updates, and the recovered state. A declared entry may still have value zero.

### Run and inspect

```sh
cmake --build build/release --target custom_response litepgo_dev
build/release/examples/energy/response/custom_response
PYTHONPATH=build/release/python uv run --no-sync python examples/energy/response/custom_response.py
```

::: details Complete Python source — `custom_response.py`

<<< ../../examples/energy/response/custom_response.py

:::

::: details Complete C++ source — `custom_response.cpp`

<<< ../../examples/energy/response/custom_response.cpp

:::


## Batched material-force fitting

**Problem.** Prescribe an affine deformation on two tetrahedra sharing a face,
with $F_p=I$, and identify shared Neo-Hookean $\theta=(\lambda,\mu)$:

$$
u(X)=(F-I)X,\quad
F=\begin{bmatrix}1.15&0.12&0\\0.03&0.94&0.08\\0&0.02&1.07\end{bmatrix},
$$

$$
f(\theta)=\nabla_uE(u,\theta),\qquad
\min_\theta\tfrac12\|f(\theta)-f(2,3)\|^2.
$$

The first equation defines displacement $u(X)$; $F$ is fixed. Thus the
Jacobian is a fixed-pose partial derivative, not an equilibrium sensitivity.
The [solid density](../guide/fem/volume.md) defines $E$.

**API walkthrough.** `ConstantDofMap(topology,2)` shares material parameters.
`VolumetricFemForceMaterialResponse[NeoHookean,TetP1Basis,TetCentroidQuadrature]`
returns integrated positive forces and their sparse Jacobian. `ForceResidual`
subtracts observations generated independently by
`VolumetricFemEnergy.eval_gradient`. Gauss–Newton starts at $(0.7,1.2)$.
Batch-size-one and larger batches use the same protocol.

**Expected result.** Both batching paths agree with the energy gradient and
with each other's sparse structure and values; the parameter Jacobian has rank
two. Both fitted parameters recover $(2,3)$. This is noise-free fixed-pose
force calibration, without an inner displacement solve.

Python optionally accepts `--copies`, `--batch-size`, and `--repeats`. C++ accepts
those quantities positionally and an optional thread limit. Disconnected copies
exercise assembly throughput. The default run performs correctness checks
without timing repetitions; timing results are machine-dependent.

### Run and inspect

```sh
cmake --build build/release --target mesh_response litepgo_dev
build/release/examples/energy/response/mesh_response
PYTHONPATH=build/release/python uv run --no-sync python examples/energy/response/mesh_response.py
```

::: details Complete Python source — `mesh_response.py`

<<< ../../examples/energy/response/mesh_response.py

:::

::: details Complete C++ source — `mesh_response.cpp`

<<< ../../examples/energy/response/mesh_response.cpp

:::
