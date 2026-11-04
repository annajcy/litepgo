# Newton and Gauss–Newton examples

These programs need no input files. They construct explicit objectives with known
solutions. The [solver guide](../../guide/solver/index.md) explains damping, feasible
steps, convergence criteria, and failure statuses.

All commands run from the repository root after [configuring the release build](../../guide/build.md). Python commands explicitly select that build. Expand the source sections for complete programs, including imports, setup, checks, and output handling.

## Examples

- [Constrained chain](./newton-chain.md). Hard constraints and a complete Newton solve.
- [TBB thread limits](./thread-limit.md). Python scoped limits and C++ runtime/arena controls.
- [Exponential curve fitting](./gauss-newton-fit.md). Analytic nonlinear least squares.
