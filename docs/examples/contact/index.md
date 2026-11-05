# Contact examples

These examples evaluate native IPC on prescribed surface positions and combine
it with Tet P1 elasticity, loading and constraints in static solves. The static
examples show the mapping between displacement unknowns and surface-position
contact inputs.

All commands run from the repository root after [configuring the release build](../../guide/build.md). Python commands explicitly select that build. Expand the source sections for complete programs, including imports, setup, checks, and output handling.

## Examples

- [Pressed tet box stack](./tet-p1-box-stack.md). Boundary mapping, native IPC, and changing sparsity.
- [Compression between penalty plates](./tet-p1-plate-compression.md). Elasticity, IPC self-contact, and two one-sided plate penalties.
- [Area weighting under mesh refinement](./area-weighting.md). Fixed parallel sheets compare contact energy and normal force with area weighting off and on.
- [Read-only active contact snapshot](./contact-snapshot.md). Export and plot every merged active record, including mollified EE stencils.
- [Three primitive groups](./pair-weights.md). Fixed group multipliers scale contact energy and force before feature merging.
