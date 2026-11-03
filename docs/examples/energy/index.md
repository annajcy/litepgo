# Scalar energies and vector responses

These four programs explain energy and response APIs before an equilibrium solve.
They generate their own data and write no meshes. A zero exit means their numerical
checks passed. The [energy/response guide](../../guide/energy/index.md) defines
sparse patterns, callback shapes, and ownership.

All commands run from the repository root after [configuring the release build](../../guide/build.md). Python commands explicitly select that build. Expand the source sections for complete programs, including imports, setup, checks, and output handling.

## Examples

- [Custom scalar energies](./custom-energy.md). Fixed/dynamic scalar derivatives and sparse structure.
- [Energy composition](./energy-composition.md). Weighted sums and affine derivative pullback.
- [Custom vector responses](./custom-response.md). Fixed/dynamic vector residuals and Jacobians.
- [Batched material-force fitting](./mesh-response.md). Batched fixed-pose material-force calibration.
