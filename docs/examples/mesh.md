# Mesh examples

The Laplacian and ARAP examples generate a 9-by-9 triangulated square (81 vertices). Unknowns are
current vertex positions, not displacements. Edge weights are uniformly one.
The programs teach custom local terms, `StencilDofMap`, batched `MeshEnergy`,
and hard handles; they are not arbitrary-mesh processing command-line tools.

All commands run from the repository root after [configuring the release build](../guide/build.md). Python commands explicitly select that build. Expand the source sections for complete programs, including imports, setup, checks, and output handling.

## Laplacian smoothing and editing

**Problem.** Let $L_{ii}=|N(i)|$, $L_{ij}=-1$ for neighbors, and
$K=L\otimes I_3$. Smoothing minimizes

$$
E_s(x)=\tfrac12x^TKx=\tfrac12\sum_{\{i,j\}}\|x_i-x_j\|^2
$$

with the whole boundary fixed at rest. Interior heights are perturbed. The
free equilibrium is the original flat grid. Editing instead preserves rest
coordinates $p$ through

$$
E_d(x)=\tfrac12\|Kx-Kp\|^2.
$$

The left edge stays at rest and the right edge is lifted by 0.5. A one-ring
residual is $r_i=\sum_j[(x_i-x_j)-(p_i-p_j)]$. With
$B_i=[|N(i)|I_3,-I_3,\ldots]$, local derivatives are $B_i^Tr_i$ and
$B_i^TB_i$. Squaring the whole one-ring sum preserves cross-neighbor terms;
the global editing Hessian is $K^TK$.

**API walkthrough.** Edge stencils `[i,j]` supply smoothing terms. One-ring
stencils `[i,*N(i)]` supply editing terms. The callback returns values `(B,)`,
gradients `(B,n)`, and stacked Hessians `(B*n,n)`. `DofReduction` selects fixed
coordinate indices; each initial position vector supplies the handle values.
Newton solves the quadratic free systems.

**Expected result.** Assertions check the flat result, exact handles, energy
reduction, and interior movement. An optional output-directory argument replaces
`output/cpp/laplacian_optimization/` or `output/python/laplacian_optimization/`.
Files are `rest.obj`, `noisy.obj`, `smoothed.obj`, `handles.obj`, and `edited.obj`;
rerunning replaces them. Uniform smoothing can shrink geometry. Editing is not
rotation invariant, and neither objective prevents collisions or inversion.

### Run and inspect

```sh
cmake --build build/release --target laplacian_optimization litepgo_python
build/release/examples/mesh/laplacian_optimization
PYTHONPATH=build/release/python uv run --no-sync python examples/mesh/laplacian_optimization.py
```

::: details Complete Python source — `laplacian_optimization.py`

<<< ../../examples/mesh/laplacian_optimization.py

:::

::: details Complete C++ source — `laplacian_optimization.cpp`

<<< ../../examples/mesh/laplacian_optimization.cpp

:::


## ARAP local-global deformation

**Problem.** Let $p_{ij}=p_i-p_j$, $d_{ij}=x_i-x_j$, and $R_i$ be
proper rotations. Minimize

$$
E(x,R)=\frac14\sum_{\{i,j\}}
\left(\|d_{ij}-R_ip_{ij}\|^2+\|d_{ij}-R_jp_{ij}\|^2\right).
$$

The local step takes $S_i=\sum_jd_{ij}p_{ij}^T=U\Sigma V^T$ and
$R_i=U\operatorname{diag}(1,1,\det(UV^T))V^T$, excluding reflections.
During the global solve rotations are fixed. With $B=[I_3,-I_3]$,
$r_i=d-R_ip$, and $r_j=d-R_jp$,

$$
g_e=\tfrac12B^T(r_i+r_j),\qquad H_e=B^TB.
$$

The assembled global Hessian is $L\otimes I_3$. Alternate until the largest
coordinate update is below $10^{-8}$, with a 1,000-iteration cap.

**API walkthrough.** An edge map drives `MeshEnergy`. Each global iteration
captures its rotation array, then calls constrained Newton. The left edge stays
at rest; the right edge is rotated/translated before those positions are fixed.
Rotations are refitted only after the global solve.

**Expected result.** Rigid-motion invariance, energy descent, exact handles,
and interior displacement are checked. Output reports iteration count, energy
decrease, and update size. An optional output-directory argument replaces
`output/cpp/arap/` or `output/python/arap/`, containing `rest.obj`, `handles.obj`,
and `edited.obj`. This is a local alternating solution, with no global-optimality,
volume-preservation, injectivity, or collision guarantee.

### Run and inspect

```sh
cmake --build build/release --target arap litepgo_python
build/release/examples/mesh/arap
PYTHONPATH=build/release/python uv run --no-sync python examples/mesh/arap.py
```

::: details Complete Python source — `arap.py`

<<< ../../examples/mesh/arap.py

:::

::: details Complete C++ source — `arap.cpp`

<<< ../../examples/mesh/arap.cpp

:::

## Constrained isotropic remeshing

Remesh a square triangulated around an off-center vertex. The target edge length
is `0.8`, so the unit boundary edges meet the protected-edge limit of
`4/3 * 0.8`. Pass input vertex ID `4` to hold the interior point at
`(0.2, 0.2, 0)`. The remesher can change interior connectivity.

Both programs find the fixed point by coordinate and compare the boundary
segments by their endpoint coordinates, since output indices may change.
They also run `has_self_intersections` on the result and print the before and
after triangle counts.

### Run and inspect

```sh
cmake --preset release -DLITEPGO_WITH_CGAL=ON
cmake --build build/release --target constrained_remesh litepgo_python
build/release/examples/mesh/constrained_remesh
PYTHONPATH=build/release/python uv run --no-sync python examples/mesh/constrained_remesh.py
```

::: details Complete Python source — `constrained_remesh.py`

<<< ../../examples/mesh/constrained_remesh.py

:::

::: details Complete C++ source — `constrained_remesh.cpp`

<<< ../../examples/mesh/constrained_remesh.cpp

:::
