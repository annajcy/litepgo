# Energy composition

[Back to Energies and responses](./index.md)

**Problem.** Combine a regularizer and target springs, then change coordinates:

$$
F(x)=\tfrac14\|x\|^2+\|x-t\|^2,\quad t=(1,2,3)^T,\quad x=Tz+c,
$$

$$
T=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix},\quad
c=(0,0,1)^T,\quad z=(0.2,-0.3)^T,
$$

$$
\nabla_zF=T^T(2.5x-2t),\qquad H_z=2.5T^TT.
$$

**API walkthrough.** `QuadraticEnergy(I)` and `VertexPullingEnergy` supply the
terms. `WeightedEnergySum` weights them by 0.5 and 2.0. `AffineMappedEnergy(total,
T, c)` performs the pullback; refresh its pattern before requesting the Hessian.

**Expected result.** The program prints energy, a two-entry gradient, and a
2-by-2 Hessian, comparing each to the equations above. No optimization occurs;
this isolates composition and the affine chain rule.

## Run and inspect

```sh
cmake --build build/release --target energy_composition litepgo_python
build/release/examples/energy/scalar/energy_composition
PYTHONPATH=build/release/python uv run --no-sync python examples/energy/scalar/energy_composition.py
```

::: details Complete Python source — `energy_composition.py`

<<< ../../../examples/energy/scalar/energy_composition.py

:::

::: details Complete C++ source — `energy_composition.cpp`

<<< ../../../examples/energy/scalar/energy_composition.cpp

:::
