# Your first complete solve

After [building litepgo](build.md), solve a chain with six scalar unknowns and
prescribed endpoints. This example needs no mesh files. It introduces the energy,
constraint, and solver objects used again in FEM and contact.

## Problem and expected solution

Let $z=(z_0,\ldots,z_5)$. Minimize

$$
E(z)=\frac12\sum_{i=0}^{4}(z_{i+1}-z_i)^2,
\qquad z_0=1,\quad z_5=3.
$$

If $D$ is the forward-difference matrix, then $E=\frac12z^TD^TDz$,
$g=D^TDz$, and $H=D^TD$. Each free row of the stationarity equation gives
$2z_i-z_{i-1}-z_{i+1}=0$. Therefore

$$
z_i=1+\frac{2i}{5},\qquad z=(1,1.4,1.8,2.2,2.6,3).
$$

The unconstrained Hessian has a constant-vector nullspace. Fixing the endpoints
removes that nullspace from the free system.

## Complete Python program

```python
import numpy as np
import scipy.sparse as sp
from litepgo import dof, energy, solver

n = 6
D = sp.diags([-np.ones(n - 1), np.ones(n - 1)], [0, 1],
             shape=(n - 1, n), format="csc")
objective = energy.QuadraticEnergy((D.T @ D).tocsc())
initial = np.zeros(n)
initial[0], initial[-1] = 1.0, 3.0
fixed = dof.DofReduction(n, [0, n - 1])
options = solver.NewtonOptions(gradient_tolerance=1e-10)
result = solver.newton_solve(objective, fixed, initial, options)
assert result.status == solver.NewtonStatus.Converged
np.testing.assert_allclose(result.x, np.linspace(1.0, 3.0, n), atol=1e-10)
print(result.x)
```

`QuadraticEnergy` stores the symmetric matrix. `DofReduction` selects fixed
indices; their values come from `initial`. `NewtonOptions.gradient_tolerance`
is required and measures the free-gradient infinity norm. The result's `x`
contains all six unknowns, including the prescribed values. Inspect `status`
before using a result as a converged equilibrium.

## Run the maintained C++ and Python versions

The maintained programs also print per-iteration diagnostics:

```sh
cmake --build build/release --target newton_chain litepgo_dev
build/release/examples/solver/newton_chain
PYTHONPATH=build/release/python uv run --no-sync python examples/solver/newton_chain.py
```

C++ supplies an explicit sparse solver, while Python defaults to
`AutoSparseSolver`. Here is the full native program:

<<< ../../examples/solver/newton_chain.cpp

## Continue

[Core concepts](concepts.md) defines packing, force signs, and sparse patterns.
[Solvers](solver/index.md) explains convergence and failure statuses.
The [solver examples](../examples/solvers.md) also introduce nonlinear least squares.
