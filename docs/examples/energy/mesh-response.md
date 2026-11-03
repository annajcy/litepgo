# Batched material-force fitting

[Back to Energies and responses](./index.md)

**Problem.** Prescribe an affine deformation on two tetrahedra sharing a face,
with $F_p=I$, and identify shared Neo-Hookean $\theta=(\lambda,\mu)$:

$$
u(X)=(F-I)X,\quad
F=\begin{bmatrix}1.15&0.12&0\\0.03&0.94&0.08\\0&0.02&1.07\end{bmatrix},
$$

$$
f(\theta)=\nabla_uE(u,\theta),\qquad
\min_\theta\tfrac12\|f(\theta)-f(2,3)\|^2.
$$

The first equation defines displacement $u(X)$; $F$ is fixed. Thus the
Jacobian is a fixed-pose partial derivative, not an equilibrium sensitivity.
The [solid density](../../guide/fem/volume.md) defines $E$.

**API walkthrough.** `ConstantDofMap(topology,2)` shares material parameters.
`VolumetricFemForceMaterialResponse[NeoHookean,TetP1Basis,TetCentroidQuadrature]`
returns integrated positive forces and their sparse Jacobian. `ForceResidual`
subtracts observations generated independently by
`VolumetricFemEnergy.eval_gradient`. Gauss–Newton starts at $(0.7,1.2)$.
Batch-size-one and larger batches use the same protocol.

**Expected result.** Both batching paths agree with the energy gradient and
with each other's sparse structure and values; the parameter Jacobian has rank
two. Both fitted parameters recover $(2,3)$. This is noise-free fixed-pose
force calibration, without an inner displacement solve.

Python optionally accepts `--copies`, `--batch-size`, and `--repeats`. C++ accepts
those quantities positionally and an optional thread limit. Disconnected copies
exercise assembly throughput. The default run performs correctness checks
without timing repetitions; timing results are machine-dependent.

## Run and inspect

```sh
cmake --build build/release --target mesh_response litepgo_python
build/release/examples/energy/response/mesh_response
PYTHONPATH=build/release/python uv run --no-sync python examples/energy/response/mesh_response.py
```

::: details Complete Python source — `mesh_response.py`

<<< ../../../examples/energy/response/mesh_response.py

:::

::: details Complete C++ source — `mesh_response.cpp`

<<< ../../../examples/energy/response/mesh_response.cpp

:::
