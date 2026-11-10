# Free shell relaxation with prescribed plastic deformation

[Back to FEM basics](./index.md)

This example applies a spatially varying, fixed plastic deformation gradient to
an open Schwarz P shell and directly minimizes its **Koiter–StVK shell energy**
starting from the rest shape. It uses only litepgo, NumPy and SciPy; plotting
additionally requires Matplotlib. It does not depend on fit_micro or a custom
material extension.

![Original and relaxed Schwarz P shell](/images/examples/shell-fp-free-relaxation.png)

Left: original geometry. Right: the converged shape, colored by displacement
magnitude. Both views use the same scale; displacement is not exaggerated.

## Model and boundary conditions

The bundled `assets/mesh/schwarz_p_open_cell.obj` is a unit-side-length Schwarz P
midsurface exported from the fit_micro experiment. Thickness is 0.06, Young's
modulus is 1e8 and Poisson's ratio is 0.35. Each original face is integrated once.
The cut cell has free open edges: **no PBC, contact energy or external forces**.
Missing neighboring faces use the mid-edge shell's open-boundary curvature rule.

At each reference face centroid, with normalized height Z,

$$
q=0.65\cos(2\pi Z),\qquad F_p=\operatorname{diag}(e^q,e^{-q},1).
$$

Thus det(Fp)=1, with prescribed principal stretches approximately 0.522–1.916.
`ShellFemEnergy` transforms the natural metric and curvature using this fixed
per-face Fp while retaining the original area weights and thickness. A spatially
varying target is generally incompatible, so its relaxed energy need not vanish.
Fp is applied at full amplitude from the beginning; there is no continuation.

A six-DOF rigid-frame gauge removes translations and rotations: vertex A is
fixed, B can move along AB, and C can move in the initial ABC plane. No triangle
edge lengths are fixed. All other vertex coordinates are free. The script checks
the **full** nodal gradient after solving, including the gauge vertices.

## Run

From the litepgo repository root, after building its Python module:

```sh
uv run --no-sync python examples/fem/material/shell_fp_free_relaxation.py
uv run --no-sync python examples/fem/material/plot_shell_fp_free_relaxation.py
```

Outputs go to `output/schwarz_p_fp_free/`:

- `original.obj`, `relaxed.obj`: identical connectivity, before/after geometry.
- `relaxed.npz`: geometry, prescribed Fp, reduced state and gauge vertex indices.
- `summary.json`: energy, residuals, iteration count and optional SPD check.
- `comparison.png`: the illustrated comparison.

A Release build on Apple M1 Pro converged in 49 Newton iterations: energy
540564.2597 → 48830.2409, maximum nodal displacement 0.19527, full gradient
infinity norm 1.91e-8. The rigid-gauge-reduced Hessian passed positive-definite
factorization. Iteration counts and rounding can vary with the linear solver.
If AccelerateLLT/PardisoLLT is unavailable, the optional SPD diagnostic is null.

The example asserts convergence, reduced energy and a full residual below 1e-5.
It does not test global optimality or self-intersections. With contact disabled,
this example demonstrates constitutive relaxation, not collision-safe deformation.

## Source

<<< ../../../examples/fem/material/shell_fp_free_relaxation.py

<<< ../../../examples/fem/material/plot_shell_fp_free_relaxation.py
