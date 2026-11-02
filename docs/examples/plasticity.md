# Equilibrium with prescribed plastic fields

These programs prescribe a plastic state and solve the displacement. They do not
integrate a yield law or infer a loading history. Volume and shell models use
different density conventions, stated below. Clamps remove rigid modes; there
are no applied loads in these examples.

For the three volume twists, let $r=(X_x-c_x,X_y-c_y)$,
$\phi(Z)=\frac\pi2(Z-Z_{\min})/H$, and $R(\phi)$ be a planar rotation.
Prescribe nodal displacement $u_p=(R(\phi)r-r,0)$, interpolate it in the chosen
displacement space, and form

$$
F_p=I+\nabla_Xu_p,\quad F=I+\nabla_Xu,\quad F_e=FF_p^{-1},
$$

$$
E(u)=\sum_{e,q}w_q\det J_X\det F_p\,\psi(F_e),
$$

$$
\psi(F_e)=\frac\mu2(\|F_e\|_F^2-3)-\mu\log\det F_e
+\frac\lambda2(\log\det F_e)^2,\qquad (\lambda,\mu)=(2,3).
$$

The spatially varying twist includes shear in $F_p$; it is not just the nodal
rotation. The smooth twist is volume-preserving, but its finite-element
interpolant need not have exactly unit determinant. Released boundary conditions
admit the compatible stress-free shape $X+u_p$.

Volume programs accept optional input VEG and base output VEG paths. Each writes
`<stem>_free` and `<stem>_clamped` VEG/OBJ, with the bottom fixed in both runs and
the top additionally fixed in the latter. C++ defaults to `output/plastic_twist/`,
Python to `output/python/plastic_twist/`. Rerunning replaces outputs. Each program
checks convergence and fixed values, then prints elastic energy, free residual,
displacement, and top twist. Expect a released top rotation near 90 degrees and
near-zero energy, while the doubly clamped state retains positive elastic energy.

All commands run from the repository root after [configuring the release build](../guide/build.md). Python commands explicitly select that build. Expand the source sections for complete programs, including imports, setup, checks, and output handling.

## Tet P1 plastic twist

**Problem and discretization.** Use `assets/box-tet.veg`, P1 geometry and
P1 plastic/displacement fields with tetrahedron-centroid integration. The program
checks the identity plastic state, then advances the prescribed twist through
four nonzero increments, warm-starting each equilibrium.

**API walkthrough.** A width-3 `VertexDofMap` gathers rest and plastic nodal
values. Build `InterpolatedField3D[TetP1Basis]` for each, then
`DeformationGradientField3D(u_p, X)` for each $F_p$. Those borrowed inputs
must stay alive. Pass the fields to `VolumetricFemEnergy`, construct fixed xyz
indices for the bottom/top faces, and call Newton on the elastic energy alone.

**Expected result.** Both runs converge; the released run matches its compatible
plastic shape while clamping the top prevents full relaxation. The program also
writes `<stem>_rest.obj`. This validates a prescribed elastic rest-state change,
not a return-mapping plasticity algorithm.

### Run and inspect

```sh
cmake --build build/release --target tet_p1_plastic_twist litepgo_python
build/release/examples/fem/plasticity/tet_p1_plastic_twist
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/plasticity/tet_p1_plastic_twist.py
```

::: details Complete Python source — `tet_p1_plastic_twist.py`

<<< ../../examples/fem/plasticity/tet_p1_plastic_twist.py

:::

::: details Complete C++ source — `tet_p1_plastic_twist.cpp`

<<< ../../examples/fem/plasticity/tet_p1_plastic_twist.cpp

:::


## Hex Q1 plastic twist

**Problem and discretization.** Use `assets/box-hex.veg`, Q1 geometry,
Q1 interpolation of $u_p$, and `GaussLegendre3D[2]`. Unlike the tet loading
loop, this program directly constructs the final 90-degree plastic twist before
its two solves.

**API walkthrough.** Use width-3 vertex maps and Q1 coordinate/plastic fields.
The composed $F_p$ is supplied to
`VolumetricFemEnergy[NeoHookean,HexQ1Basis,GaussLegendre3D[2]]`.
`DofReduction` fixes bottom xyz entries, then bottom and top xyz entries for the
second solve. The same material snapshot is used for both.

**Expected result.** The displacement space contains the prescribed Q1
$u_p$, so the released state can be stress-free. The clamped energy is
positive. Exported VEG and OBJ display corner-interpolated geometry; they do not
establish a continuum injectivity guarantee between integration points.

### Run and inspect

```sh
cmake --build build/release --target hex_q1_plastic_twist litepgo_python
build/release/examples/fem/plasticity/hex_q1_plastic_twist
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/plasticity/hex_q1_plastic_twist.py
```

::: details Complete Python source — `hex_q1_plastic_twist.py`

<<< ../../examples/fem/plasticity/hex_q1_plastic_twist.py

:::

::: details Complete C++ source — `hex_q1_plastic_twist.cpp`

<<< ../../examples/fem/plasticity/hex_q1_plastic_twist.cpp

:::


## Hex Hermite plastic twist

**Problem and discretization.** On `assets/box-hex.veg`, supply the twist's
values and physical derivatives in all eight Hermite vector slots. Q1 rest
geometry and `GaussLegendre3D[4]` integration are used. Both equilibrium and the
compatible plastic displacement belong to the same Hermite space.

**API walkthrough.** `VertexDofMap(topology,24)` gathers physical DOFs.
`BasisDofTransform[HexHermiteBasis]` converts plastic slots to reference
coefficients before constructing `DeformationGradientField3D`. End clamps fix
slots 0, 1, 2, and 4: $u,u_{,X},u_{,Y},u_{,XY}$, which determine a face
at constant $Z$. Fixing value slots alone would not fix that face field.

For surface output, `sample_boundary[HexQ1Basis](...,4)` generates reference
samples and `build_point_displacement_map[HexHermiteBasis]` evaluates the solved
field there. The OBJ therefore includes derivative-DOF contributions between
corners, with 32 triangles per boundary quad.

**Expected result.** Released and clamped equilibria preserve fixed physical
slots. The free case relaxes toward the compatible plastic shape. VEG stores
only displaced corners; use the sampled OBJ to inspect the Hermite boundary.
Rest boxes must be positive, affine, and axis-aligned. A Q1 plastic interpolant
would generally be incompatible with C1 Hermite displacement across faces.

### Run and inspect

```sh
cmake --build build/release --target hex_hermite_plastic_twist litepgo_python
build/release/examples/fem/plasticity/hex_hermite_plastic_twist
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/plasticity/hex_hermite_plastic_twist.py
```

::: details Complete Python source — `hex_hermite_plastic_twist.py`

<<< ../../examples/fem/plasticity/hex_hermite_plastic_twist.py

:::

::: details Complete C++ source — `hex_hermite_plastic_twist.cpp`

<<< ../../examples/fem/plasticity/hex_hermite_plastic_twist.cpp

:::


## Differential shell growth

**Problem.** Generate a 4-by-1 xy strip with 65 vertices and 96 triangles.
Clamp its first two vertex columns, fixing both position and the strip direction.
With face-centroid coordinate $y_e$, prescribe

$$
F_{p,e}(\alpha)=\operatorname{diag}(e^{2\alpha y_e},1,1),\qquad
\alpha:0\longrightarrow0.04.
$$

The two sides acquire different natural in-plane lengths. For each original
stencil, define $Y_i=F_{p,e}(X_i-X_0)$, take natural forms from $Y$, and
minimize the [Koiter shell energy](../guide/fem/shell.md) using original area
weights. Thickness is 0.2 and $(\lambda_{ps},\mu)=(2,3)$. There is no volume
$\det F_p$ factor or plastic-thickness update in this convention.

**API walkthrough.** Build generated topology, mid-edge stencils, the stencil
map, and rest data. Pass per-face `Fp` as the final `ShellFemEnergy[KoiterStVK]`
argument. Reconstruct the energy for each prescribed field, check the identity
state, and warm-start four growth increments. Only displacement is solved.

**Expected result.** An in-plane curved strip with nonzero displacement, fixed
clamp values, and small free-force residuals. The reported residual here is a
Euclidean norm, while Newton stops using its max norm. The optional argument is
an output directory, defaulting to `output/shell_plastic_field/` in C++ and
`output/python/shell_plastic_field/` in Python. Files are `rest.obj` and
`equilibrium.obj`. A planar natural stencil stays zero-curvature under its local
affine transform; this example does not prescribe an out-of-plane buckling branch.

### Run and inspect

```sh
cmake --build build/release --target shell_plastic_field litepgo_python
build/release/examples/fem/plasticity/shell_plastic_field
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/plasticity/shell_plastic_field.py
```

::: details Complete Python source — `shell_plastic_field.py`

<<< ../../examples/fem/plasticity/shell_plastic_field.py

:::

::: details Complete C++ source — `shell_plastic_field.cpp`

<<< ../../examples/fem/plasticity/shell_plastic_field.cpp

:::
