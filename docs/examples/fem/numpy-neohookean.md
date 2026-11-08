# A volumetric material written in NumPy

[Back to FEM basics](./index.md)

`NumpyNeoHookean` subclasses `fem.PyVolumetricMaterial` and implements the full
material contract with vectorized, closed-form NumPy derivatives. Native FEM uses
it on the bundled hexahedral box with Hermite displacement (24 DOFs per vertex)
and 4×4×4 Gauss points: each batch of up to 256 elements becomes one `eval` call
with 64 points per element. This Python-only example has no C++ counterpart.

With $G=F_p^{-1}$, $d=\det F_p$, $F_e=FG$, $J=\det F_e$ and
$c=\lambda\log J-\mu$,

$$
W=d\left[\frac\mu2(\|F_e\|_F^2-3)-\mu\log J+\frac\lambda2(\log J)^2\right],\qquad
S=\frac{\partial\psi}{\partial F_e}=\mu F_e+cF_e^{-T},\qquad
P=\frac{\partial W}{\partial F}=d\,SG^T.
$$

With $C_{abmn}=\partial^2\psi/\partial F_{e,ab}\partial F_{e,mn}
=\mu\delta_{am}\delta_{bn}+\lambda F^{-1}_{e,ba}F^{-1}_{e,nm}-cF^{-1}_{e,bm}F^{-1}_{e,na}$,
the code evaluates

$$
\frac{\partial P_{ij}}{\partial F_{kl}}=d\,C_{ibkn}G_{ln}G_{jb},\qquad
\frac{\partial P_{ij}}{\partial (F_p)_{kl}}
=d\left[(SG^T)_{ij}G_{lk}-C_{ibmn}F_{e,mk}G_{ln}G_{jb}-S_{ib}G_{jk}G_{lb}\right]
$$

with `numpy.einsum`, and $\partial P/\partial\lambda=d\log J\,F_e^{-T}G^T$,
$\partial P/\partial\mu=d(F_e-F_e^{-T})G^T$. Each output is computed only when its
name is in `outputs`. Outside $\det F_e>0$ the logarithm is NaN, which rejects
the evaluation; no feasible-step hook is needed because the default keeps
$\det F>0$.

`main` compares the energy, gradient and Hessian with native `NeoHookean`, then
both force-response Jacobians (whose mixed derivatives come from the formulas
above), and finally solves the corner-pull equilibrium of the
[Hermite example](../equilibrium/hex-hermite-corner-pull.md) with each material.
The relative displacement difference is at roundoff level: the Hermite derivative
DOFs reach large values, and parallel sums round differently from run to run.
2×2×2 Gauss points would under-integrate Hermite hexes and admit zero-energy
modes. See [materials written in Python](../../guide/fem/python-materials.md).

## Run and inspect

```sh
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/material/numpy_neohookean.py
```

::: details Complete Python source — `numpy_neohookean.py`

<<< ../../../examples/fem/material/numpy_neohookean.py

:::
