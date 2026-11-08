# Pressed tet box stack

[Back to Contact](./index.md)

Contact uses the default fixed reference-area weighting. Its coefficient is
set directly on the raw barrier energy; no normalization conversion is applied.

**Geometry and boundary conditions.** Load `assets/box-tet.veg` and duplicate it
in the y direction. If one box has height $H$, separate them by $0.04H$
and offset the upper box laterally by $(0.013H,0,0.017H)$ to avoid an exactly
symmetric bifurcation. Clamp the lower box's bottom xyz DOFs. Pull the upper top
face toward displacement $(0,-0.2H,0)$, with unit spring stiffness.

**Formulation.** Both bodies use Neo-Hookean $(\lambda,\mu)=(2,3)$,
$F_p=I$, Tet P1 interpolation and centroid integration. Let $W$ select
boundary displacement and $p=X_s+Wu$. Minimize

$$
\Pi(u)=E_{\rm solid}(u)+\tfrac12\sum_{i\in\mathcal T}\|u_i-(0,-0.2H,0)\|^2
+10^3 C(p).
$$

The surface barrier uses $\hat d=0.01H$, $d_{\min}=0$, and candidate terms

$$
b(d^2,\hat d^2)=-(d^2-\hat d^2)^2\log(d^2/\hat d^2)
\quad\text{for }0<d<\hat d,
$$

with zero outside the activation distance and the native edge mollifier where
needed. The [contact guide](../../guide/contact/index.md) specifies candidates, multiplicity,
mollification, and CCD. Surface derivatives pull back as
$W^Tg_p$ and $W^TH_pW$.

**API walkthrough.** Construct `VolumetricFemEnergy` on the combined topology.
Use `extract_surface_with_vertices` and `build_vertex_displacement_map` to retain
exact volume IDs. `SurfaceKinematics` holds $X_s,W$.
`IPCContactEnergy(surface, options, SpatialHash())` has unit stiffness;
`AffineMappedEnergy` converts it to volume displacement coordinates, and
`WeightedSumEnergy` applies the $10^3$ weight. `DofReduction` fixes the bottom.
Newton manages contact's changing Hessian pattern and feasible-step bounds.

**Expected result.** A converged solve with a small free-gradient residual,
positive active pair count, and $0<d_{\rm min,active}<\hat d$.
Call contact diagnostics on `kinematics.positions(result.x)`, not directly on
volume displacement. The program returns failure if convergence or touching
checks fail and prints elastic, pulling, contact, residual, and gap diagnostics.

Optional positional arguments are input VEG and output VEG. Defaults write
`output/tet_p1_box_stack.veg` and `.obj` in C++, and
`output/python/tet_p1_box_stack.*` in Python. Files are replaced on rerun.
The model has no friction, dynamics, plastic evolution, or adaptive stiffness.
Feasible floating-point contact handling has the numerical limits described in
the guide; it is not an exact-arithmetic robustness guarantee.

## Run and inspect

```sh
cmake --build build/release --target tet_p1_box_stack litepgo_python
build/release/examples/contact/tet_p1_box_stack
PYTHONPATH=build/release/python uv run --no-sync python examples/contact/tet_p1_box_stack.py
```

::: details Complete Python source — `tet_p1_box_stack.py`

<<< ../../../examples/contact/tet_p1_box_stack.py

:::

::: details Complete C++ source — `tet_p1_box_stack.cpp`

<<< ../../../examples/contact/tet_p1_box_stack.cpp

:::
