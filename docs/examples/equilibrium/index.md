# Elastic equilibrium with soft corner loads

The four examples minimize total potential from zero displacement:

$$
\Pi(u)=E_{\rm elastic}(u)+\frac{k}{2}\sum_{i\in\mathcal C}\|u_i-t_i\|^2.
$$

Here $\mathcal C$ is the pulled-corner set and $t_i$ is an anchor displacement,
not an absolute position. The springs remove rigid motions; the examples use
`DofReduction(num_dofs, [])`, with no hard constraints. `WeightedEnergySum`
combines elastic and spring energies; Newton uses projected element Hessians,
feasible steps, and free-gradient tolerance $10^{-8}$.

For solids, $F=I+\nabla_Xu$, $F_p=I$, $(\lambda,\mu)=(2,3)$, and

$$
E_{\rm elastic}=\sum_{e,q}w_q\det J_X\,
\left[\frac\mu2(\|F\|_F^2-3)-\mu\log\det F
+\frac\lambda2(\log\det F)^2\right].
$$

All four accept optional input and output paths, in that order. Defaults use
bundled assets. C++ writes `output/<name>.veg` (or `.obj` for shell), Python
`output/python/<name>.*`; volume cases also write boundary OBJ beside VEG.
Rerunning replaces these files. Exit status reports convergence. Printed potential,
elastic/pulling energies, free residual, and corner-to-anchor gap help interpret
soft-load equilibrium; the gap need not be zero.

All commands run from the repository root after [configuring the release build](../../guide/build.md). Python commands explicitly select that build. Expand the source sections for complete programs, including imports, setup, checks, and output handling.

## Examples

- [Tet P1 corner pull](./tet-p1-corner-pull.md). Tetrahedral elastic equilibrium.
- [Hex Q1 corner pull](./hex-q1-corner-pull.md). Trilinear hex equilibrium.
- [Hex Hermite corner pull](./hex-hermite-corner-pull.md). Physical derivative DOFs and high-order displacement.
- [Shell saddle pull](./shell-saddle-pull.md). Membrane/bending energy with soft corner loads.
