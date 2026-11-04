# Scalar energies and vector responses

These programs explain energy and response APIs, assembly, and a small equilibrium solve.
They generate their own data and write no meshes. The runnable programs show API calls and report results; independent analytic
checks live in the test suite. The [energy/response guide](../../guide/energy/index.md) defines
sparse patterns, batch hook shapes, and ownership.

All commands run from the repository root after [configuring the release build](../../guide/build.md). Python commands explicitly select that build. Expand the source sections for complete programs, including imports, setup, checks, and output handling.

## Examples

- [Custom scalar energies](./custom-energy.md). Fixed/dynamic scalar derivatives and sparse structure.
- [Mesh energy subclass](./mesh-energy-subclass.md). Batch overrides, shared-node assembly, and fixed-end spring equilibrium.
- [Energy composition](./energy-composition.md). Weighted sums and affine derivative pullback.
- [Mesh response subclass](./mesh-response-subclass.md). Subclass assembly, Jacobian products, and parameter fitting.
- [Custom vector responses](./custom-response.md). Fixed/dynamic vector residuals and Jacobians.
- [Batched material-force fitting](./mesh-response.md). Batched fixed-pose material-force calibration.
