# Constrained chain

[Back to Newton and Gauss–Newton](./index.md)

**Problem.** Minimize

$$
\min_z\frac12\sum_{i=0}^4(z_{i+1}-z_i)^2,\qquad z_0=1,\quad z_5=3.
$$

If $D$ takes forward differences, $g=D^TDz$ and $H=D^TD$. The free
stationarity equations give equally spaced values.

**API walkthrough.** Build `QuadraticEnergy(D.T @ D)`.
`DofReduction(6,[0,5])` fixes indices; `initial` supplies endpoint values 1 and
3. Pass `convergence=AbsoluteConvergence(1e-10)` to `newton_solve`, and inspect
`NewtonStatus.Converged`. Its callback reports the free-gradient max norm and
accepted step; a terminal callback has step zero.

**Expected result.** `(1,1.4,1.8,2.2,2.6,3)`, with exactly preserved endpoints.
The full Hessian has a translation nullspace that disappears from the constrained
free system. Both implementations assert the solution and print iteration data.
The [quickstart](../../guide/quickstart.md) derives the equations.

## Run and inspect

```sh
cmake --build build/release --target newton_chain litepgo_python
build/release/examples/solver/newton_chain
PYTHONPATH=build/release/python uv run --no-sync python examples/solver/newton_chain.py
```

::: details Complete Python source — `newton_chain.py`

<<< ../../../examples/solver/newton_chain.py

:::

::: details Complete C++ source — `newton_chain.cpp`

<<< ../../../examples/solver/newton_chain.cpp

:::
