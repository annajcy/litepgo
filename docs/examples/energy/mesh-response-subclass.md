# Mesh response subclass

[Back to Energies and responses](./index.md)

This example defines a `LinearResidual` subclass of `MeshResponse`, with
model state and overrides for evaluation, JVP, and VJP. See also
[global vector responses](./custom-response.md).

## Batch evaluation and assembly

Three elements share two input parameters. Each element returns two local
residuals, with output stencils `(0,1)`, `(1,2)`, and `(0,2)`.
Contributions to shared output indices are **summed**, not concatenated.

Each local residual is `A @ theta - observed[e]`, where

$$
A=\begin{pmatrix}1&2\\3&-1\end{pmatrix},
\qquad \theta_{\mathrm{true}}=(2,3).
$$

The observations are generated at the known parameters. The independently
assembled global Jacobian is rectangular:

$$
J=\begin{pmatrix}2&4\\4&1\\6&-2\end{pmatrix}.
$$

The constructor is
`MeshResponse(input_map, output_map, batch_size=2)`.
The hook receives original element IDs, local inputs `(B,2)`, and a
Jacobian request. Python returns values `(B,2)` alone, or
`(values, Jacobians)` with Jacobians `(2B,2)`: one vertically stacked
local matrix per element. C++ fills the Jacobian pointer only when non-null.

There are three elements and batches of at most two, so the example also covers
a final partial batch. Select observations using `ids`; hook execution
order is unspecified and batches may run concurrently. Captured data stay fixed.

## Products and fitting

Both programs check values and the explicit Jacobian against the independent
global formula. They also verify JVP/VJP, the adjoint identity
`w @ Jv == v @ JTw`, and a directional finite difference.

The subclass overrides `eval_jvp_batch` as `dx @ A.T` and
`eval_vjp_batch` as `gy @ A`, avoiding even local Jacobian construction.
Omitting these overrides uses the base defaults, which form local Jacobians,
multiply locally, and scatter without assembling a global numerical Jacobian.
See [batched assembly](../../guide/energy/assembly.md).

Finally, the existing explicit Gauss–Newton solver recovers `(2,3)`.
This is a small algebraic hook example, not an equilibrium or constitutive
model. Both programs print:

```text
mesh response subclass: shared assembly, JVP/VJP and parameter fit verified
```

## Run and inspect

```sh
cmake --build build/release --target mesh_response_subclass litepgo_python
build/release/examples/energy/response/mesh_response_subclass
PYTHONPATH=build/release/python uv run --no-sync python examples/energy/response/mesh_response_subclass.py
```

::: details Complete Python source — `mesh_response_subclass.py`

<<< ../../../examples/energy/response/mesh_response_subclass.py

:::

::: details Complete C++ source — `mesh_response_subclass.cpp`

<<< ../../../examples/energy/response/mesh_response_subclass.cpp

:::
