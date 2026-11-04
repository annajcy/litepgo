# Mesh energy subclass

[Back to Energies and responses](./index.md)

**Problem.** Connect four scalar node positions with three springs. The edge
stencils are `(0,1)`, `(1,2)`, `(2,3)`, stiffnesses are $(2,3,4)$, and rest
lengths are $(1,2,1)$. Fix $x_0=0$ and $x_3=7$, and minimize

$$
E(x)=\sum_{e=0}^{2}\frac{k_e}{2}(x_{e+1}-x_e-\ell_e)^2.
$$

**API walkthrough.** Define `SpringEnergy : MeshEnergy` in C++ or
`SpringEnergy(energy.MeshEnergy)` in Python. The constructor passes the map and
batch size to the base and stores stiffnesses and rest lengths as model state.
Override `eval_batch` to supply the spring formulas. The assembler sums
contributions at shared nodes.

Each hook receives original element `ids`, local positions shaped `(B,2)`,
and requests for gradient/Hessian outputs. For stretch $s=x_1-x_0-\ell_e$,
return the local value $k_es^2/2$, gradient $k_es(-1,1)$, and Hessian

$$
H_e=k_e\begin{pmatrix}1&-1\\-1&1\end{pmatrix}.
$$

Python returns `(values, gradients, Hessians)` with shapes `(B,)`, `(B,2)`,
and `(2B,2)`; unused derivatives are `None`. C++ fills requested output pointers.
The example uses `batch_size=2`, so the final batch has only one edge. Select
spring data using `ids`, never an assumed hook order. The model arrays
remain fixed while hooks run, including concurrent batches.

The energy is polynomial and has no restricted domain, so it needs no feasible
step hook. `newton_solve` handles fixed endpoints through `DofReduction`.
For larger models using the same subclass API,
see the [Laplacian](../mesh/laplacian-optimization.md) and [ARAP](../mesh/arap.md)
examples for that style.

**Checks and expected result.** Both programs independently verify global energy,
gradient, and Hessian using an incidence matrix. Equal tension in springs in
series gives $t=36/13$ and equilibrium positions

$$
x=(0,31/13,69/13,7).
$$

They verify these positions, unchanged endpoints, and zero free-node gradient.
Both print `mesh energy subclass: 3 springs, shared-node assembly and fixed-end equilibrium verified`.
No mesh assets or output files are required.

## Run and inspect

```sh
cmake --build build/release --target mesh_energy_subclass litepgo_python
build/release/examples/energy/scalar/mesh_energy_subclass
PYTHONPATH=build/release/python uv run --no-sync python examples/energy/scalar/mesh_energy_subclass.py
```

::: details Complete Python source — `mesh_energy_subclass.py`

<<< ../../../examples/energy/scalar/mesh_energy_subclass.py

:::

::: details Complete C++ source — `mesh_energy_subclass.cpp`

<<< ../../../examples/energy/scalar/mesh_energy_subclass.cpp

:::
