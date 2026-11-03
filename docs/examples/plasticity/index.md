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

All commands run from the repository root after [configuring the release build](../../guide/build.md). Python commands explicitly select that build. Expand the source sections for complete programs, including imports, setup, checks, and output handling.

## Examples

- [Tet P1 plastic twist](./tet-p1-plastic-twist.md). Compatible plastic fields and incremental equilibrium.
- [Hex Q1 plastic twist](./hex-q1-plastic-twist.md). Q1 plastic rest-state change.
- [Hex Hermite plastic twist](./hex-hermite-plastic-twist.md). C1 plastic fields and sampled Hermite output.
- [Differential shell growth](./shell-plastic-field.md). Incompatible per-face natural geometry.
