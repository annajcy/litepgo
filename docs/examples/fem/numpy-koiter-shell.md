# A shell material written in NumPy

[Back to FEM basics](./index.md)

`NumpyKoiter` subclasses `fem.PyShellMaterial` and reproduces `KoiterStVK`
with closed-form NumPy derivatives, including the mixed derivative with respect to
natural forms used by the plastic force response. This Python-only example has no
C++ counterpart.

The material receives packed forms $s=(a_{11},a_{12},a_{22},b_{11},b_{12},b_{22})$.
The off-diagonal entry is one independent variable, so each energy term is a
quadratic form in a packed 3-vector. With $A=\bar a$, symmetric basis matrices
$E_1=\begin{bmatrix}1&0\\0&0\end{bmatrix}$,
$E_2=\begin{bmatrix}0&1\\1&0\end{bmatrix}$,
$E_3=\begin{bmatrix}0&0\\0&1\end{bmatrix}$,
$l_r=\operatorname{tr}(A^{-1}E_r)$ and $T_{rs}=\operatorname{tr}(A^{-1}E_rA^{-1}E_s)$,

$$
\psi(S)=\frac{\lambda_{ps}}2[\operatorname{tr}(A^{-1}S)]^2+\mu\operatorname{tr}(A^{-1}SA^{-1}S)
=\tfrac12\,s^TQs,\qquad Q=\lambda_{ps}\,ll^T+2\mu T.
$$

With $e=\tfrac12(a-\bar a)$ and $k=b-\bar b$ in packed form,
$W=\tfrac t2e^TQe+\tfrac{t^3}{24}k^TQk$, so $\partial W/\partial a=\tfrac t2Qe$,
$\partial W/\partial b=\tfrac{t^3}{12}Qk$, and the Hessian blocks are
$\tfrac t4Q$ and $\tfrac{t^3}{12}Q$. The natural-form derivative uses
$\partial A^{-1}/\partial\bar a_c=-A^{-1}E_cA^{-1}$, giving
$\partial l_r/\partial\bar a_c=-T_{cr}$ and
$\partial T_{rs}/\partial\bar a_c=-(R_{crs}+R_{rcs})$ with
$R_{xyz}=\operatorname{tr}(A^{-1}E_xA^{-1}E_yA^{-1}E_z)$.

`main` compares the energy, gradient and Hessian with native `KoiterStVK` on a
slightly curved nine-node patch with per-face plastic matrices, then the material
and plastic force-response Jacobians, and finally solves an equilibrium with the
boundary prescribed and the center node free. See
[materials written in Python](../../guide/fem/python-materials.md#shell-materials).

## Run and inspect

```sh
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/material/numpy_koiter_shell.py
```

::: details Complete Python source — `numpy_koiter_shell.py`

<<< ../../../examples/fem/material/numpy_koiter_shell.py

:::
