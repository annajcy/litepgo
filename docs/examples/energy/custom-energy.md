# Custom scalar energies

[Back to Energies and responses](./index.md)

**Problem.** For $x\in\mathbb R^2$, evaluate

$$
E_0(x)=\tfrac12\|x-(1,2)\|^2,\qquad
E_1(x)=\tfrac12\|x\|^2+\tfrac14[\max(0,x_0+x_1)]^4.
$$

With $s=\max(0,x_0+x_1)$ and $\mathbf1=(1,1)^T$,

$$
\nabla E_0=x-(1,2),\quad H_0=I,\qquad
\nabla E_1=x+s^3\mathbf1,\quad H_1=I+3s^2\mathbf1\mathbf1^T.
$$

**API walkthrough.** `Quadratic` derives from `FixedPatternEnergy` with diagonal
structure. `Coupled` derives from `DynamicPatternEnergy`; `pattern_at` returns
full structure when $x_0+x_1>0$, diagonal otherwise. Both implement
`eval_values`, returning requested derivatives and `None` for unused outputs.
Both energies are convex, so Exact and Projected Hessians can coincide here.

Refresh structure with `update_pattern` before Hessian assembly. Value-only
calls do not need a refresh. C++ receives output pointers; Python returns owning
arrays. A second refresh at the same state reports no structural change.

**Expected result.** At $(1,1)$, $E_0=0.5$, $E_1=5$, and
$\nabla E_1=(9,9)$. The stored Hessian count follows `2 -> 4 -> 2` as the
active branch changes. Printed values and assertions check both derivatives
and structure.

## Run and inspect

```sh
cmake --build build/release --target custom_energy litepgo_python
build/release/examples/energy/scalar/custom_energy
PYTHONPATH=build/release/python uv run --no-sync python examples/energy/scalar/custom_energy.py
```

::: details Complete Python source — `custom_energy.py`

<<< ../../../examples/energy/scalar/custom_energy.py

:::

::: details Complete C++ source — `custom_energy.cpp`

<<< ../../../examples/energy/scalar/custom_energy.cpp

:::
