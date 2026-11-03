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

The automatic names are compile-time selections. In C++, the linked backend
target supplies the feature macro; linking only `litepgo::linalg` selects Eigen.
The Python extension exposes matching aliases for its configured platform backend:

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

## CMake targets

| Target | Provides | Availability |
| --- | --- | --- |
| `litepgo::linalg` | Sparse patterns, `SparseSolver`, `EigenLDLT`, `EigenSparseLU` | Always |
| `litepgo::linalg_accelerate` | `AccelerateLLT`, `AccelerateLDLT`; links `linalg` | macOS with `LITEPGO_WITH_ACCELERATE=ON` |
| `litepgo::linalg_mkl` | `PardisoLLT`, `PardisoLDLT`; links `linalg` | x86-64 Linux/Windows with `LITEPGO_WITH_MKL=ON` |

For a standalone Eigen solve:

```cmake
target_link_libraries(my_app PRIVATE litepgo::linalg)
```

To use PARDISO, link `litepgo::linalg_mkl` instead; on macOS link
`litepgo::linalg_accelerate` for Accelerate. Backend implementation sources and
vendor libraries stay separate from the base `linalg` target.

The [solver module](../solver/index.md) links the configured platform backend,
preserving its automatic defaults. It manages analysis, factorization, damping,
and retries for its own nonlinear iterations. `litepgo::core` also includes that
backend through `solver`.
