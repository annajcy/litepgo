# Newton and Gauss–Newton examples

These programs need no input files. They construct explicit objectives with known
solutions. The [solver guide](../guide/solver/index.md) explains damping, feasible
steps, convergence criteria, and failure statuses.

All commands run from the repository root after [configuring the release build](../guide/build.md). Python commands explicitly select that build. Expand the source sections for complete programs, including imports, setup, checks, and output handling.

## Constrained chain

**Problem.** Minimize

$$
\min_z\frac12\sum_{i=0}^4(z_{i+1}-z_i)^2,\qquad z_0=1,\quad z_5=3.
$$

If $D$ takes forward differences, $g=D^TDz$ and $H=D^TD$. The free
stationarity equations give equally spaced values.

**API walkthrough.** Build `QuadraticEnergy(D.T @ D)`.
`DofReduction(6,[0,5])` fixes indices; `initial` supplies endpoint values 1 and
3. Set `NewtonOptions.gradient_tolerance=1e-10`, call `newton_solve`, and inspect
`NewtonStatus.Converged`. Its callback reports the free-gradient max norm and
accepted step; a terminal callback has step zero.

**Expected result.** `(1,1.4,1.8,2.2,2.6,3)`, with exactly preserved endpoints.
The full Hessian has a translation nullspace that disappears from the constrained
free system. Both implementations assert the solution and print iteration data.
The [quickstart](../guide/quickstart.md) derives the equations.

### Run and inspect

```sh
cmake --build build/release --target newton_chain litepgo_dev
build/release/examples/solver/newton_chain
PYTHONPATH=build/release/python uv run --no-sync python examples/solver/newton_chain.py
```

::: details Complete Python source — `newton_chain.py`

<<< ../../examples/solver/newton_chain.py

:::

::: details Complete C++ source — `newton_chain.cpp`

<<< ../../examples/solver/newton_chain.cpp

:::


## Exponential curve fitting

**Problem.** Fit amplitude $a$ and rate $b$ to nine samples at
$t_i\in[0,2]$:

$$
y_i=2.5e^{-0.8t_i},\quad r_i(a,b)=ae^{-bt_i}-y_i,\quad
\min_{a,b}\tfrac12\sum_i r_i^2,
$$

$$
\partial_a r_i=e^{-bt_i},\qquad \partial_b r_i=-at_i e^{-bt_i}.
$$

**API walkthrough.** A `FixedPatternResponse` has a 9-by-2 structural pattern.
Its evaluator skips Jacobian work when `need_jacobian` is false. The solver
starts at `(1,0.3)`, uses damped sparse normal equations, and stops on
$\|J^Tr\|_\infty$.

**Expected result.** `(2.5,0.8)` and objective below $10^{-16}$. Output includes
parameters, objective, and iteration count. There are no parameter bounds in
this example. Noisy data need not have zero residual when stationary; parameter
scaling affects conditioning.

### Run and inspect

```sh
cmake --build build/release --target gauss_newton_fit litepgo_dev
build/release/examples/solver/gauss_newton_fit
PYTHONPATH=build/release/python uv run --no-sync python examples/solver/gauss_newton_fit.py
```

::: details Complete Python source — `gauss_newton_fit.py`

<<< ../../examples/solver/gauss_newton_fit.py

:::

::: details Complete C++ source — `gauss_newton_fit.cpp`

<<< ../../examples/solver/gauss_newton_fit.cpp

:::
