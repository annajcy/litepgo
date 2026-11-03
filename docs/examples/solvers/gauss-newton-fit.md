# Exponential curve fitting

[Back to Newton and Gauss–Newton](./index.md)

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

## Run and inspect

```sh
cmake --build build/release --target gauss_newton_fit litepgo_python
build/release/examples/solver/gauss_newton_fit
PYTHONPATH=build/release/python uv run --no-sync python examples/solver/gauss_newton_fit.py
```

::: details Complete Python source — `gauss_newton_fit.py`

<<< ../../../examples/solver/gauss_newton_fit.py

:::

::: details Complete C++ source — `gauss_newton_fit.cpp`

<<< ../../../examples/solver/gauss_newton_fit.cpp

:::
