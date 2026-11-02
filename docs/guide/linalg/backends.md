# Sparse backend selection

Choose a backend for the matrix structure and definiteness of your problem.
All backends use the same [factorization lifecycle](factorization.md).

## Choose a backend

| Backend | Intended system |
| --- | --- |
| `EigenLDLT` | Symmetric positive-definite systems |
| `EigenSparseLU` | General sparse systems |
| `AccelerateLLT` / `AccelerateLDLT` | macOS positive-definite / indefinite systems, if enabled |
| `PardisoLLT` / `PardisoLDLT` | oneMKL positive-definite / indefinite systems, if enabled |

The automatic names are compile-time selections, exposed as matching Python
aliases:

| Compiled feature | `AutoSparseSolver` | `AutoIndefiniteSparseSolver` |
| --- | --- | --- |
| Accelerate | `AccelerateLLT` | `AccelerateLDLT` |
| Otherwise, oneMKL | `PardisoLLT` | `PardisoLDLT` |
| Neither | `EigenLDLT` | `EigenSparseLU` |

Use `litepgo.build_info()` and module exports to inspect the active build.
oneMKL/PARDISO is enabled by default on x86-64 Linux and Windows; macOS uses
Accelerate. Disable `LITEPGO_WITH_MKL` to select the Eigen defaults instead.
Automatic selection does not make a singular or badly scaled system solvable.
For example, projecting local Hessians can help a forward Newton step, but
[equilibrium sensitivities](../solver/equilibrium.md#differentiate-an-equilibrium-branch)
require the exact unshifted free stiffness and an appropriate backend.

Link `litepgo::linalg` in C++. The [solver module](../solver/index.md) manages analysis,
factorization, damping, and retries for its own nonlinear iterations.
