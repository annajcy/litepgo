# Laplacian mesh optimization

[`laplacian_optimization.cpp`](laplacian_optimization.cpp) and its
[Python mirror](laplacian_optimization.py) solve two classical geometry-processing
problems using `StencilDofMap`, `MeshEnergy` (C++ and Python), `DofReduction`, and Newton. They generate the
same 9 × 9 triangulated square, so no asset download is needed.

## Objectives

Each unique undirected triangle edge has weight one. The symmetric graph
Laplacian has `L_ii = degree(i)` and `L_ij = -1` for neighboring vertices.
With vertex-major xyz coordinates, `K = L ⊗ I₃`.

1. **Fixed-boundary Laplacian smoothing:** minimize
   `E(x) = 0.5 xᵀ K x = 0.5 Σ_edges ||x_i - x_j||²`, holding the entire boundary
   at its original planar positions. Interior vertices start with deterministic
   height perturbations. At equilibrium, each free vertex is the average of its
   neighbors. This regular triangulation recovers the original flat grid.
2. **Differential-coordinate mesh editing:** minimize
   `E(x) = 0.5 ||K x - K x_rest||²`. Fix the left edge and prescribe a height of
   0.5 on the right edge; the top and bottom interiors remain free. Use
   one stencil `[i, *N(i)]` per vertex, with local residual
   `r_i = Σ_j (x_i - x_j) - Σ_j (p_i - p_j)` and energy `0.5 ||r_i||²`.
   Its local Jacobian is `B_i = [degree(i) I₃, -I₃, ...]`; the callback returns
   `B_iᵀ r_i` and `B_iᵀ B_i`. Assembly produces the global Hessian `Kᵀ K`.
   The full energy, including its constant, is retained.

Both problems are quadratic with positive-definite free Hessians after fixing
these handles, so a full Newton step solves them up to numerical precision.
The examples check convergence, exact fixed coordinates, expected planar
coordinates, objective/residual reduction, and movement of an interior vertex.
C++ checks remain active in Release builds.

## Run

From the repository root, after configuring the release preset:

```bash
cmake --build build/release --target laplacian_optimization _core
build/release/examples/mesh/laplacian_optimization
PYTHONPATH=build/release/python .venv/bin/python examples/mesh/laplacian_optimization.py
```

Outputs go to `output/cpp/laplacian_optimization` and
`output/python/laplacian_optimization`. Either program accepts an optional output
directory as its sole argument. Re-running replaces the five named OBJ files:

| File | Contents |
| --- | --- |
| `rest.obj` | Original flat grid |
| `noisy.obj` | Smoothing input with interior height perturbations |
| `smoothed.obj` | Fixed-boundary smoothing result |
| `handles.obj` | Editing input with the right handle lifted |
| `edited.obj` | Editing result after interior vertices relax |

## Scope

These use the uniform graph Laplacian, not cotangent or mass-normalized weights.
Smoothing minimizes edge lengths and can shrink a general mesh; it is not a
volume-preserving denoiser. Differential coordinates are preserved in the global
frame, so the editing objective is not rotation invariant (unlike ARAP).
Neither objective prevents triangle inversion or self-intersection on general
inputs. This example deliberately uses a generated grid and prescribed vertical
motion; it is not an arbitrary-mesh processing CLI.

## ARAP deformation (C++ / Python)

[`arap.cpp`](arap.cpp) and its [Python mirror](arap.py) implement uniform-weight vertex ARAP with local/global
iterations. It fixes the left edge and moves the right edge of the generated grid.
The local step uses SVD with determinant correction to fit proper rotations;
the global step assembles edge callbacks with `MeshEnergy` and uses constrained Newton.
For each undirected edge, the energy is
`0.25 (||x_i-x_j-R_i(p_i-p_j)||² + ||x_i-x_j-R_j(p_i-p_j)||²)`.
Each edge stencil has two vertices. With `B = [I₃, -I₃]`, the callback returns
`0.25 (||r_i||² + ||r_j||²)`, gradient `Bᵀ (r_i + r_j)/2`, and Hessian `Bᵀ B`.
Rotations stay fixed during each solve. Assembly produces global Hessian `L ⊗ I₃`;
the right-hand side uses the average of the endpoint rotations. Outer convergence requires a maximum coordinate update below
`1e-8`, with a hard cap of 1000 iterations. This is a local solution, not a global
optimality guarantee. The example checks rigid-motion invariance, energy descent,
exact handles, and interior movement.

```bash
cmake --build build/release --target arap
build/release/examples/mesh/arap
PYTHONPATH=build/release/python .venv/bin/python examples/mesh/arap.py
```

Writes `rest.obj`, `handles.obj`, and `edited.obj` under `output/cpp/arap` and `output/python/arap`, or
an optional output directory. Re-running replaces those files. C++ numerical checks remain active in Release builds. Uniform weights are positive; no cotangent weights,
collision barriers, or inversion constraints are included. ARAP preserves local
shape approximately and does not guarantee volume preservation or injectivity.

## Local terms and assembly API

Notation: `p_i` is the original position, `x_i` the optimized position, `N(i)`
the neighbors, and every edge has symmetric weight one. Connectivity stays fixed.
All three problems impose hard handles `x_i = h_i` through `DofReduction`:
the map stores fixed indices and the initial solver vector supplies their values.

| Problem | Stencil | Local energy |
| --- | --- | --- |
| Smoothing | `[i, j]`, each undirected edge once | `0.5 ||x_i - x_j||²` |
| Editing | `[i, *N(i)]`, once per vertex | `0.5 ||(LX)_i - (LP)_i||²` |
| ARAP global step | `[i, j]`, each undirected edge once | `0.25 (||d - R_i p||² + ||d - R_j p||²)` |

For ARAP, `d = x_i-x_j` and `p = p_i-p_j`. This is exactly the directed-neighbor
sum with prefactor `1/4`; both endpoint rotations contribute. Editing squares the
whole one-ring sum, retaining cross terms between neighbors.

Both C++ and Python use the same batch-only MeshEnergy protocol.

## MeshEnergy API

C++ includes `litepgo/energy/scalar/mesh_energy.hpp`:

```cpp
MeshEnergy energy(map, eval_batch, 256);
MeshEnergy bounded(map, eval_batch, 256, step_batch);
```

The callback `eval_batch(ids, x, gradient*, hessian*)` returns a Vec of
per-element energies. Inputs are row-major matrices with shape (B, n);
gradients have shape (B, n), Hessians (B*n, n), stacked in element order.
Requested outputs must be fully overwritten; either pointer can be null.

Python uses `MeshEnergy(map, eval_batch, batch_size=256, step=None)`.
The callback `eval_batch(ids, x, need_gradient, need_hessian)` returns
(values, gradients, hessians) with the same shapes as C++; unrequested
derivatives may be None. Python inputs own their storage and may be retained.
Reshape x explicitly inside the callback if a vertex/component layout is useful.

Elements are grouped by DOF count. TBB concurrently evaluates batches, then
EnergyAssembler assembles derivatives in deterministic element order. Callbacks
must safely share captured state and must not depend on invocation order.
Python callbacks acquire the GIL; parallel speedup depends on work that releases it.
Projection is per element and writes back into each batch Hessian block.
MeshEnergy reuses batch storage; EnergyAssembler reads non-owning strided views
directly, without repacking derivatives into per-element matrices.
Batch size 1 preserves all batch dimensions and uses the same execution path.

Optional step(ids, x, dx) returns B finite bounds in [0,1]; the global bound
is their minimum. Step callbacks can also run concurrently.
Maps are snapshotted at construction. Batch size does not cap total assembly memory.

The separate BatchedMeshEnergy class and scalar-element callback were removed.
