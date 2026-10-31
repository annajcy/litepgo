# Energy and solver examples

Each example has matching C++ and Python implementations, requires no input files,
prints its result, and checks it numerically. C++ checks remain enabled in Release builds.

| Example | What it demonstrates | Expected result |
| --- | --- | --- |
| [`energy_composition`](../energy/energy_composition.cpp) / [Python](../energy/energy_composition.py) | `QuadraticEnergy`, `VertexPullingEnergy`, `WeightedEnergySum`, and `AffineMappedEnergy`; value, gradient, sparse Hessian | Direct agreement with the quadratic formula and chain rule |
| [`newton_chain`](newton_chain.cpp) / [Python](newton_chain.py) | `DofReduction`, Newton tolerance, iteration callback, and convergence status | Six scalar displacements evenly spaced from 1 to 3 |
| [`gauss_newton_fit`](gauss_newton_fit.cpp) / [Python](gauss_newton_fit.py) | Dense nonlinear least squares with an analytic Jacobian; residual-only trial evaluations | Amplitude 2.5 and decay rate 0.8 |

From the repository root, after configuring the release preset:

```bash
cmake --build build/release --target energy_composition newton_chain gauss_newton_fit
build/release/example/energy/energy_composition
build/release/example/solver/newton_chain
build/release/example/solver/gauss_newton_fit

cmake --build build/release --target _core
PYTHONPATH=build/release/python .venv/bin/python example/energy/energy_composition.py
PYTHONPATH=build/release/python .venv/bin/python example/solver/newton_chain.py
PYTHONPATH=build/release/python .venv/bin/python example/solver/gauss_newton_fit.py
```

The energy example evaluates `F(T z + offset)` with
`F(x) = 0.25 ||x||² + ||x - target||²`. Its gradient is
`Tᵀ (2.5 x - 2 target)` and its Hessian is `2.5 Tᵀ T`.
C++ calls `update_pattern` and copies `hessian_pattern` before filling the Hessian;
Python allocates the output matrix through `eval`. These terms have fixed sparsity.

The Newton example minimizes `0.5 sum_i (x[i+1] - x[i])²`. The full Hessian has a
translation nullspace, but fixing both endpoints makes the free system positive
definite. Fixed values come from the initial vector. The required
`gradient_tolerance` applies only to free DOFs; the callback also receives the
terminal state, whose step length is zero.

Gauss–Newton minimizes `0.5 ||r||²`, with `r_i = a exp(-b t_i) - y_i` and Jacobian
columns `exp(-b t)` and `-a t exp(-b t)`. The data are noiseless so the known
parameters provide a self-check. Convergence is based on `||Jᵀ r||∞`, which does
not imply a zero residual for general data. This example does not impose parameter
bounds. For a larger application with equilibrium sensitivities, see
[`material_fit`](../fem/material_fit.py).
