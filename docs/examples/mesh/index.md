# Mesh optimization examples

The Laplacian and ARAP examples start from a 9-by-9 triangulated square by
default. The Python versions also accept tetrahedral input. The unknowns are
current vertex positions, and edge weights are uniformly one.
The programs teach custom local terms, `StencilDofMap`, batched `MeshEnergy`,
and hard handles; they are not arbitrary-mesh processing command-line tools.

All commands run from the repository root after [configuring the release build](../../guide/build.md). Python commands explicitly select that build. Expand the source sections for complete programs, including imports, setup, checks, and output handling.

## Examples

- [Laplacian smoothing and editing](./laplacian-optimization.md). Edge/one-ring assembly, smoothing and editing.
- [ARAP local-global deformation](./arap.md). Rotation fitting and constrained global solves.
