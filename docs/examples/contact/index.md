# Contact examples

These examples combine Tet P1 elasticity, soft loading, hard constraints, surface
extraction, and a native IPC barrier in static solves. They show the mapping
between displacement unknowns and surface-position contact inputs.

All commands run from the repository root after [configuring the release build](../../guide/build.md). Python commands explicitly select that build. Expand the source sections for complete programs, including imports, setup, checks, and output handling.

## Examples

- [Pressed tet box stack](./tet-p1-box-stack.md). Boundary mapping, native IPC, and changing sparsity.
- [Compression between penalty plates](./tet-p1-plate-compression.md). Elasticity, IPC self-contact, and two one-sided plate penalties.
