# Custom vector responses

[Back to Energies and responses](./index.md)

**Problem.** Evaluate and solve linear and branch-dependent residuals:

$$
r_0(x)=\begin{bmatrix}x_0-1\\x_1-2\\x_0+x_1-3\end{bmatrix},\quad
J_0=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix},
$$

$$
r_1(x)=\begin{bmatrix}x_0-1\\x_1+\max(0,x_0)^2-2\end{bmatrix},\quad
J_1=\begin{bmatrix}1&0\\2\max(0,x_0)&1\end{bmatrix}.
$$

**API walkthrough.** Derive the linear class from `FixedPatternResponse` and the
nonlinear class from `DynamicPatternResponse`. Implement `eval_impl` to return
`r` or `(r, J)` according to `need_jacobian`. The dynamic class adds/removes the
lower-left entry through `jacobian_pattern_at`. Ordinary Jacobian evaluation
automatically prepares the pattern for each input. The C++ example also shows
explicit preparation and `_into` buffer reuse. Pass the residual object to `gauss_newton_solve`; its pattern
is refreshed automatically at current iterates.

**Expected result.** The linear residual vanishes at $(1,2)$, with four
Jacobian entries. The nonlinear solve goes from $(-1,-1)$ to $(1,1)$,
with three active Jacobian entries. `tests/python/test_api_examples.py` checks
dimensions, structural updates, and the recovered state. A declared entry may still have value zero.

## Run and inspect

```sh
cmake --build build/release --target custom_response litepgo_python
build/release/examples/energy/response/custom_response
PYTHONPATH=build/release/python uv run --no-sync python examples/energy/response/custom_response.py
```

::: details Complete Python source — `custom_response.py`

<<< ../../../examples/energy/response/custom_response.py

:::

::: details Complete C++ source — `custom_response.cpp`

<<< ../../../examples/energy/response/custom_response.cpp

:::
