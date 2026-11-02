# Energy and Response subclass APIs

Both scalar energies and vector responses live in `litepgo::energy` (C++) and
`litepgo.energy` (Python). Include scalar headers from `litepgo/energy/scalar/` and response headers from
`litepgo/energy/response/`.
The former `litepgo::response`, `litepgo.response`, and `litepgo/response/` paths
are no longer provided; class names and sparsity-pattern behavior are unchanged.

Two matching C++/Python pairs demonstrate the base protocols without mesh files:

| Example | Demonstration |
| --- | --- |
| [custom_energy.cpp](scalar/custom_energy.cpp) / [Python](scalar/custom_energy.py) | Inherit `FixedPatternEnergy` and `DynamicPatternEnergy`; scalar value, gradient, sparse Hessian; structural growth and shrinkage |
| [custom_response.cpp](response/custom_response.cpp) / [Python](response/custom_response.py) | Inherit `FixedPatternResponse` and `DynamicPatternResponse`; rectangular Jacobian, dynamic structure, Gauss-Newton |

```sh
cmake --build build/debug --target custom_energy custom_response litepgo_dev
build/debug/examples/energy/scalar/custom_energy
build/debug/examples/energy/response/custom_response
uv run --no-sync python examples/energy/scalar/custom_energy.py
uv run --no-sync python examples/energy/response/custom_response.py
ctest --test-dir build/debug -R '^(EnergyApiExample|ResponseApiExample)$' --output-on-failure
```

Energy represents a scalar `E(x)`. Its C++ override fills independently optional
gradient and Hessian buffers. Python `eval_values(x, want_gradient, hessian_kind)`
returns `(value, gradient_or_None, Hessian_or_None)`. The example energies are
convex, so their Exact and Projected Hessians coincide; the binding does not
perform projection for a custom energy. An optional `feasible_step(x, dx)`
override is needed for restricted domains; these examples are feasible everywhere.

Response represents a vector `y(x)`. Python `eval_values(x, need_jacobian)` returns
`y` or `(y, J)`; C++ writes requested Jacobian values into the supplied sparse
matrix. The fixed example has three outputs and two inputs. Gauss-Newton treats
the vector as a residual targeting zero: the caller must subtract observations
and apply measurement weights in the response.

Dynamic subclasses implement `pattern_at(x)`. Matrix dimensions stay fixed;
stored values are ignored. Call `update_pattern(x)` before requesting derivatives.
C++ refreshes the derivative buffer from `hessian_pattern()` or
`jacobian_pattern()` after a change. Python allocates the buffer through `eval`
or derivative convenience methods. Value-only calls do not require a pattern
update. Newton and Gauss-Newton manage updates automatically.

The dynamic energy adds `max(x0+x1,0)^4/4`; the dynamic response adds
`max(x0,0)^2` to its second component. These choices give continuous derivatives
of the required order across activation. A changing pattern alone does not make
an arbitrary piecewise function differentiable. Both implementations overwrite
inactive entries with zero and reject active entries outside the current pattern.

All examples perform numerical self-checks and return nonzero on failure. The
C++ checks remain active in Release builds; Python checks use normal assertions.

## MeshResponse

`MeshResponse(input_map, output_map, eval_batch, batch_size=256)` inherits
FixedPatternResponse. Both maps describe the same elements; repeated local indices and shared output DOFs
accumulate. The maps are snapshotted at construction.

Python callback `eval_batch(ids, x, need_jacobian)` returns values or
(values, J). Inputs are (B, ni), values (B, no), and J is (B*no, ni).
C++ uses the same row-major layout and an optional Jacobian pointer.
Batch size 1 preserves these dimensions. No basis transform, target subtraction,
or equilibrium solve is performed by MeshResponse.

All combinations of ConstantDofMap, ElementDofMap, VertexDofMap and StencilDofMap
are supported. Elements are grouped by both input and output widths.
TBB evaluates batches concurrently, then ResponseAssembler reduces disjoint
output rows in original element order. Callbacks must safely share captures
and cannot assume invocation order. Python callbacks acquire the GIL; work must
release it to benefit from concurrent execution. Malformed or nonfinite results
raise errors. The separate batched class was removed.

## Batched force fitting (`mesh_response.cpp` / `mesh_response.py`)

Both examples now use the public FEM API to fit shared Neo-Hookean parameters
`theta = [lambda, mu]` on two tetrahedra sharing a face. Geometry, displacement
and Fp are fixed; no equilibrium solve is involved. Observations come independently
from `VolumetricFemEnergy::eval_gradient`, and Gauss–Newton recovers the parameters.
The examples compare batch size 1 against the requested size (default 256).

```python
Force = fem.VolumetricFemForceMaterialResponse[
    fem.NeoHookean, fem.TetP1Basis, fem.TetCentroidQuadrature]
force = Force(dof.ConstantDofMap(topology, 2), displacement_map,
              rest_position_fields, displacement, Fp_fields, batch_size=256)
value, B = force.eval(theta, need_jacobian=True)
```

Use `ElementDofMap(topology, 2)` instead for per-element `[lambda, mu]` blocks.
The class builds identity parameter fields, integrates
`VolumetricMaterialForceLocalResponse3D`, then delegates to `MeshResponse`
and `ResponseAssembler`. There are no handwritten constitutive formulas in these
examples. A small `FixedPatternResponse` subtracts observations before solving.
The current mesh wrapper supports TetP1 only. Displacements are copied; owned/shared
geometry and Fp fields must remain unchanged during use, just as for MeshEnergy.

For custom kernels, the underlying general-purpose batch API is unchanged:

The C++ callback is
`BatchMatrix(span<const int> ids, const BatchMatrix& inputs, BatchMatrix* J)`.
`BatchMatrix` is a row-major dense matrix. Inputs have shape `(B, ni)` in input
DOF-index order, values `(B, no)`, and Jacobians `(B*no, ni)` with one block per
returned row. This **2D stacked layout** follows `MeshEnergy`. No Jacobian
is needed when the pointer is null / `need_jacobian` is false. Maps with variable
stencil sizes are grouped by **both** input and output widths. Callbacks receive
original element IDs and must not assume contiguous IDs. Batch callbacks run
concurrently. MeshResponse owns a ResponseAssembler. The
assembler reads non-owning strided element views directly from ordinary vectors
or batch matrices. TBB tasks own disjoint output rows, including all Jacobian
entries on those rows; there are no atomics or shared writes. Contributions are
summed in original element order, regardless of batch grouping or thread count.
This provides parallel work even for a Jacobian with only two columns. All local
results are retained until assembly finishes, so scratch memory scales with the
total local output/Jacobian size, not just the batch size.

Run the self-checking examples:

```sh
build/native/examples/energy/response/mesh_response
PYTHONPATH=build/native/python .venv/bin/python examples/energy/response/mesh_response.py
```

Benchmark 20,000 elements (10,000 disconnected copies), with 3 warmups and 21
timed evaluations per mode. Construction and fitting
are excluded. Reported times are medians; correctness checks run before timing.
C++ reuses its sparse output buffer; Python includes allocation/conversion at the
public API boundary. Compare batch size 1 vs N **within each language**, not
absolute C++ vs Python times.

```sh
build/native/examples/energy/response/mesh_response 10000 256 21
PYTHONPATH=build/native/python .venv/bin/python examples/energy/response/mesh_response.py --copies 10000 --batch-size 256 --repeats 21
```

The current benchmark includes constitutive evaluation and quadrature, not
precomputed coefficient multiplication. Previous timings below are historical
results from the old hand-coded StVK example and do not apply to this FEM wrapper.

Historical serial-assembly baseline (old precomputed StVK example, before ResponseAssembler/TBB), measured on Apple M1 Pro, macOS 27.0.1, Release build (2026-10-31), using the
commands above and 20,000 elements:

| Language / batch size | Value: ordinary → batch (ms) | Value+J: ordinary → batch (ms) | Value+J speedup |
| --- | ---: | ---: | ---: |
| C++ / 256 | 1.079 → 0.386 | 2.037 → 1.138 | 1.79x |
| Python / 64 | 30.593 → 3.266 | 51.487 → 4.222 | 12.20x |
| Python / 256 | 30.725 → 2.477 | 51.810 → 3.312 | 15.64x |
| Python / 1024 | 30.317 → 2.242 | 52.018 → 3.156 | 16.48x |

These are single-machine timing samples, not performance guarantees. Python's
larger gain comes from amortizing Python callback overhead and evaluating local
forces with NumPy batches. The Python Response trampoline also merges canonical
sparse columns in linear time when checking returned Jacobians, avoiding a
quadratic scan for the force-residual wrapper on large meshes.

### ResponseAssembler / TBB update (historical StVK benchmark)

`dof::ResponseAssembler` is separate from `dof::EnergyAssembler`. Construct it
with `(num_outputs, num_inputs, output_indices, input_indices)` or with
`(input_map, output_map)`. Pass one `ElementView(value[, Jacobian])` per original
element to `assemble(views, global_value, optional_global_Jacobian)`. Views borrow
buffers, including strided rows/blocks; keep those buffers alive and disjoint
from outputs. Validation finishes before output writes. `assemble` overwrites
numeric results and preserves the fixed sparse pattern. Invalid local data throws
`invalid_argument`; a non-finite assembled sum throws `domain_error`.

The C++ example accepts an optional fourth argument limiting TBB parallelism:

```sh
build/native/examples/energy/response/mesh_response 10000 256 21 1
build/native/examples/energy/response/mesh_response 10000 256 21 4
build/native/examples/energy/response/mesh_response 10000 256 21 8
```

On the same machine and workload as above, with the new implementation:

| TBB thread limit | C++ ordinary value+J (ms) | C++ batch value+J (ms) |
| --- | ---: | ---: |
| 1 | 2.845 | 2.175 |
| 4 | 2.116 | 1.555 |
| 8 | 2.065 | 1.544 |

These are full evaluations, not isolated assembly timings. Eight-thread batch
evaluation is about 1.41x faster than this implementation limited to one thread,
but is slower than the historical streaming serial batch implementation (1.138
ms). Retaining all local results, creating views and validating them adds overhead
for this cheap kernel. Python with default TBB parallelism measured 58.034 ms
ordinary vs 3.706 ms batched. Parallel assembly is therefore not an end-to-end
speedup guarantee. These measurements predate the unified API and concurrent
batch evaluation; they are historical results, not timings of the current code.


LocalResponse now provides the same pre-sized output-view path as LocalEnergy:
`accumulate_into` and `VolumetricElement3D::eval_into`. The material force wrapper
writes directly into batch value/Jacobian blocks. MeshResponse reuses batch inputs
and Jacobians and updates its borrowed element views inside the batch tasks.

Both assemblers accept repeated local indices and sum every mapped entry.
Requested local NaN/Inf, bad sizes, or wrong sparse structure raise
`invalid_argument` before output writes. Overflow of finite contributions raises
`domain_error` with unspecified outputs; a later valid call overwrites them.
Unrequested derivatives are ignored. Views must stay alive and not alias outputs.
Python MeshEnergy and MeshResponse callbacks may both return None to signal an
infeasible batch, raising ValueError.
