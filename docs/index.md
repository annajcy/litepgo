# litepgo documentation

litepgo builds numerical models from meshes, degrees of freedom, scalar energies,
and vector responses. Its C++ library and Python package support static elasticity,
shell bending, contact, geometric editing, and inverse problems.

A **scalar energy** is minimized by Newton. A **vector response** describes
measurements or residuals and can be fitted by Gauss–Newton. Finite elements and
contact supply these objects; they do not require a separate simulation framework.

## Start here

1. [Build and verify the environment](guide/build.md).
2. [Solve a complete constrained problem](guide/quickstart.md).
3. [Understand the architecture and module boundaries](guide/architecture.md).
4. [Learn the common data and derivative conventions](guide/concepts.md).
5. [Learn energy and response with a spring chain](tutorials/energy/index.md).
6. [Learn Torch materials step by step](tutorials/torch/index.md).
7. [Choose a runnable example](examples.md).
8. [Measure performance](benchmarks/index.md).

Every command assumes the repository root unless a page says otherwise. Examples
use the release preset and bundled or generated geometry. Python arrays use
`float64`; integer topology arrays use `int32` in the examples.

## Learn a model

| Task | Read |
| --- | --- |
| Understand the system and module dependencies | [Architecture](guide/architecture.md) |
| Inspect shared types and compiled capabilities | [Common](guide/common.md) |
| Store meshes and read/write geometry | [Mesh and I/O](guide/mesh/index.md) |
| Gather unknowns and impose hard constraints | [DOF maps](guide/dof/index.md) |
| Factor sparse systems and select a backend | [Linear algebra](guide/linalg/index.md) |
| Define energies, responses, and batched assembly | [Energy and response contracts](guide/energy/index.md) |
| Understand the finite-element layers | [FEM module](guide/fem/index.md) |
| Choose shape functions and prepare coefficient slots | [Basis functions](guide/fem/basis.md) |
| Choose integration points and reference weights | [Quadrature](guide/fem/quadrature.md) |
| Construct interpolated fields and derivatives | [Fields and quadrature](guide/fem/fields.md) |
| Model 3D elasticity and prescribed plasticity | [Volumetric FEM](guide/fem/volume.md) |
| Model membrane strain and bending | [Shell FEM](guide/fem/shell.md) |
| Use physical derivative DOFs | [Hex Hermite](guide/fem/hermite.md) |
| Solve equilibrium and least squares | [Solvers](guide/solver/index.md) |
| Query box overlap | [Spatial](guide/spatial.md) |
| Embed points or sample a boundary | [FEM embedding](guide/fem/embedding.md) |
| Couple surface contact to volume unknowns | [Contact](guide/contact/index.md) |
| Differentiate equilibrium and identify parameters | [Inverse problems](guide/solver/inverse.md) |
| Connect native derivatives to autograd | [Torch adapters](guide/torch.md) |

## Look up a symbol or develop the library

[API references](reference.md) are generated from the compiled Python package and
C++ headers. They complement the manual's formulations and usage explanations.
[Development](development.md) explains CMake targets, tests, bindings, and the
documentation toolchain.

The built site has separate **C++ reference** and **Python reference** entries.
Their contents depend on the source headers and selected Python build, respectively.
