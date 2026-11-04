# Custom mesh response

[Back to Energies and responses](./index.md)

This example uses a function (Python) or lambda (C++) with
`CustomMeshResponse`. For named models implemented through inheritance, see
[custom vector responses](./custom-response.md).

## Callback and assembly

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
`CustomMeshResponse(input_map, output_map, eval_batch, batch_size=2)`.
The callback receives original element IDs, local inputs `(B,2)`, and a
Jacobian request. Python returns values `(B,2)` alone, or
`(values, Jacobians)` with Jacobians `(2B,2)`: one vertically stacked
local matrix per element. C++ fills the Jacobian pointer only when non-null.

There are three elements and batches of at most two, so the example also covers
a final partial batch. Select observations using `ids`; callback execution
order is unspecified and batches may run concurrently. Captured data stay fixed.

## Products and fitting

Both programs check values and the explicit Jacobian against the independent
global formula. They also verify JVP/VJP, the adjoint identity
`w @ Jv == v @ JTw`, and a directional finite difference.

The adapter inherits the default mesh product path: form local Jacobians,
multiply locally, and scatter the products without assembling a global numerical
Jacobian. This example does not implement custom product hooks. To avoid local
Jacobians as well, subclass `MeshResponse` and override
`eval_jvp_batch` / `eval_vjp_batch`; see [batched assembly](../../guide/energy/assembly.md).

Finally, the existing explicit Gauss–Newton solver recovers `(2,3)`.
This is a small algebraic callback example, not an equilibrium or constitutive
model. Both programs print:

```text
custom mesh response: shared assembly, JVP/VJP and parameter fit verified
```

## Run and inspect

```sh
cmake --build build/release --target custom_mesh_response litepgo_python
build/release/examples/energy/response/custom_mesh_response
PYTHONPATH=build/release/python uv run --no-sync python examples/energy/response/custom_mesh_response.py
```

::: details Complete Python source — `custom_mesh_response.py`

<<< ../../../examples/energy/response/custom_mesh_response.py

:::

::: details Complete C++ source — `custom_mesh_response.cpp`

<<< ../../../examples/energy/response/custom_mesh_response.cpp

:::
