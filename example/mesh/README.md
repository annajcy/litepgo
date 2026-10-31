# Laplacian mesh optimization

[`laplacian_optimization.cpp`](laplacian_optimization.cpp) and its
[Python mirror](laplacian_optimization.py) solve two classical geometry-processing
problems using `StencilDofMap`, `MeshEnergy` (C++) / `BatchedMeshEnergy` (Python), `DofReduction`, and Newton. They generate the
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
build/release/example/mesh/laplacian_optimization
PYTHONPATH=build/release/python .venv/bin/python example/mesh/laplacian_optimization.py
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
build/release/example/mesh/arap
PYTHONPATH=build/release/python .venv/bin/python example/mesh/arap.py
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

C++ examples construct `MeshEnergy(map, eval_local)` and keep the straightforward
single-term formulas. Python examples use `BatchedMeshEnergy(map, eval_batch,
batch_size=256)` and NumPy operations across each batch. Both assemble identical
local terms through the existing assembler.

## BatchedMeshEnergy API

C++ includes `litepgo/energy/batched_mesh_energy.hpp`:

```cpp
BatchedMeshEnergy energy(map, eval_batch, 256);
BatchedMeshEnergy bounded(map, eval_batch, 256, step_batch);
```

`map` is a `StencilDofMap`. `eval_batch(ids, x, gradient*, hessian*)` returns a
`Vec` of per-term values. `ids` is `std::span<const int>`; `x` is a row-major
`BatchMatrix` with shape `B × m`. Requested gradient outputs have shape `B × m`;
Hessians have shape `(B*m) × m`, one contiguous block of `m` rows per term.
Pointers are independently nullable. Resize and overwrite requested outputs.
C++ input references are valid only during the callback.

Python:

```python
potential = BatchedMeshEnergy(mapping, eval_batch, batch_size=256, step=None)

def eval_batch(ids, x, need_gradient, need_hessian):
    # ids: (B,), original term IDs
    # x: (B,k,d), m = k*d
    return values, gradients, hessians
```

Return shapes are `(B,)`, `(B,m)`, `(B,m,m)`; unrequested derivatives may be
`None`. Noncontiguous and broadcast arrays are accepted. Inputs own their storage
and may be retained. Every term returns its full energy including weights and
constants. Derivative order is vertex-major, regardless of Eigen storage order.

Construction groups stencils by length and splits groups into at most `batch_size`
terms. IDs remain the original term IDs. Batches execute serially; assembly uses
original term order, and projected Hessians are projected per term, not per batch.
Malformed shapes, missing requested derivatives and nonfinite outputs raise errors.
Callbacks must not mutate captured parameters during evaluation.

Optional `step(ids, x, dx)` returns `(B,)` safe inclusive bounds in `[0,1]`;
the C++ version takes two batch matrices and returns a `Vec`. The global bound
is their minimum. Reject infeasible starting states. Omit the callback only when
the energy is feasible along every finite segment.

The initial implementation uses `MeshEnergy` internally to reuse validation,
projection, and assembly. This retains per-term derivative buffers and performs
an additional gather/copy pass; `batch_size` does not cap total assembly memory.
There is no Hessian or factorization cache. The original Python `MeshEnergy`
callback remains available for prototyping, but these examples use batched calls
to avoid one Python invocation per term.
