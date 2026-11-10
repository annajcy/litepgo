# Shell plasticity from an interpolated displacement field

[Back to FEM basics](./index.md)

This example turns a prescribed plastic displacement into fixed per-face natural
forms, then freely relaxes a Schwarz P shell. It compares Q1 and Hermite
interpolation of the same analytic target. Neither field prescribes the shell's
actual vertex motion.

## 1. Problem and notation

| Symbol | Meaning | Unknown in relaxation? |
|---|---|---|
| $X$ | Reference position | No |
| $u_p^\star(X)$ | Analytic target plastic displacement | No |
| $\tilde u_p(X)$ | Its Q1 or Hermite interpolant | No |
| $F_{p,f}$ | Interpolated plastic gradient sampled on face $f$ | No |
| $u$ | Actual shell vertex displacements | **Yes** |

The bounding box is an interpolation domain, not a mechanical boundary
condition. All shell vertices are free. There are no pinned vertices, PBC,
contact terms, external forces or continuation. Both solves start with $u=0$.
Material: native Koiter–StVK, thickness $t=0.06$, Young's modulus $Y=10^8$,
Poisson's ratio $\nu=0.35$.

## 2. Analytic plastic displacement

Let $X_c$ be the box center, $L_z$ its height, and $r=X-X_c$. Define

$$
\theta(Z)=\frac{\pi}{3}\frac{Z-Z_c}{L_z},\qquad
u_p^\star(X)=[R_z(\theta(Z))-I](X-X_c).
$$

Here $R_z$ is rotation about z. The bottom rotates -30 degrees and the top
+30 degrees: a total 60-degree twist. Writing $k=\pi/(3L_z)$,
$c=\cos\theta$ and $s=\sin\theta$, the exact physical gradient is

$$
F_p^\star=I+\nabla_Xu_p^\star=
\begin{bmatrix}
c&-s&k(-s r_x-c r_y)\\
s& c&k(c r_x-s r_y)\\
0&0&1
\end{bmatrix},\qquad \det F_p^\star=1.
$$

The third column contains the shear induced by the height-dependent rotation;
this gradient is not just a rotation matrix.

## 3. Corner sampling and interpolation

The geometric map is the axis-aligned bounding box:

$$X(\xi)=X_{\min}+\operatorname{diag}(L_x,L_y,L_z)\xi,
\qquad \xi\in[0,1]^3.$$

It is represented by `InterpolatedResponse[HexQ1Basis]`, with corner ordering
from `coefficient_info`. Both plastic interpolants use this same geometry.

- **Q1:** sample $u_p^\star$ at eight corners: 24 scalar DOFs.
- **Hermite:** sample values plus physical derivatives $u_X,u_Y,u_Z,u_{XY},
  u_{XZ},u_{YZ},u_{XYZ}$ at each corner: 192 scalar DOFs. Eight displacement
  vectors alone cannot determine this field. The xy and xyz derivatives vanish
  analytically for this twist; other derivatives are evaluated analytically.

`BasisDofTransform` converts physical jets into reference-basis coefficients,
including the appropriate box-length factors. For either basis,

$$\tilde u_p(X(\xi))=\sum_i N_i(\xi)c_i.$$

Q1 and Hermite agree at corner values but generally differ inside the box.

## 4. Sample Fp at reference face centroids

For original face $f=(i,j,k)$, let

$$X_f=(X_i+X_j+X_k)/3,\qquad
\xi_f=\operatorname{diag}(L_x,L_y,L_z)^{-1}(X_f-X_{\min}).$$

The assigned matrix is

$$F_{p,f}=I+
\left.\frac{\partial\tilde u_p}{\partial\xi}
\left(\frac{\partial X}{\partial\xi}\right)^{-1}\right|_{\xi_f}.$$

```python
Fp_field = fem.DeformationGradientResponse(up_field, geometry)
Fp = Fp_field.evaluate(xi_f).value.reshape(3, 3, order="F")
```

The response includes the identity and the physical-coordinate conversion.
Its matrix packing is column-major. We interpolate **displacement**, then
differentiate; we do not interpolate independently specified Fp values.
Each face receives one constant matrix, sampled once from the reference mesh.
Neither the sampling locations nor Fp change during relaxation.

## 5. Natural forms and shell minimization

For each face, apply its fixed matrix to its complete reference stencil:

$$X^p_{f,i}=F_{p,f}(X_{f,i}-X_{f,0}).$$

This includes neighboring vertices needed for bending. From this transformed
stencil, the native mid-edge kinematics computes natural metric $\bar a_f^p$
and natural curvature $\bar b_f^p$. Original integration area $A_f^0$ and
thickness are retained. The actual unknown is the displacement of every shell
vertex:

$$\min_u E(u)=\sum_f A_f^0\,
W\bigl(a_f(X+u),b_f(X+u);\bar a_f^p,\bar b_f^p,t,\lambda_{ps},\mu\bigr).$$

For clarity, the implemented Koiter–StVK density is

$$
W=t\,\psi\!\left(\frac{a-\bar a^p}{2}\right)
+\frac{t^3}{12}\psi(b-\bar b^p),
\quad
\psi(S)=\frac{\lambda_{ps}}2[\operatorname{tr}(GS)]^2
+\mu\operatorname{tr}(GSGS),\quad G=(\bar a^p)^{-1},
$$

with $\lambda_{ps}=Y\nu/(1-\nu^2)$ and $\mu=Y/[2(1+\nu)]$.
This describes the shell's natural-form construction, not a separate volumetric
plastic evolution law. The solver can shift a linear system to handle rigid
zero modes; no pinning energy or position constraint is introduced.

There is **no requirement that $u=\tilde u_p$**. For a nonlinear field,
transforming every local stencil with one centroid gradient differs from mapping
its vertices through $X+\tilde u_p(X)$. Natural curvature is affected as well.
Therefore that mapped vertex shape, exported as a field target, is not an exact
zero-energy oracle. A uniform affine field is the special compatible check.

## 6. Visualize the prescribed field

![Plastic displacement and bounding box](/images/examples/shell-plastic-displacement.png)

Gray dashed: original box. Blue arrows: $\tilde u_p$ at true scale. Orange:
box edges mapped through $X+\tilde u_p(X)$. Upper row: eight corner values and
faint reference shell. Lower row: interior samples. Left: Q1; right: Hermite.
Hermite edges can curve. These are prescribed-field visualizations, **not shell
relaxation results or box constraints**.

```sh
uv run --no-sync python examples/fem/material/plot_plastic_displacement.py
```

## 7. Relaxation results and validation

![Q1 and Hermite free relaxation](/images/examples/shell-interpolated-fp.png)

These show raw converged geometry, with no post-solve alignment or displacement
magnification. Colors show actual vertex displacement magnitude.

| Interpolant | Iterations | Final energy | Full nodal residual | Sampled det(Fp) range |
|---|---:|---:|---:|---:|
| Q1 | 22 | 3848.40 | 5.00e-9 | 0.7500–0.9927 |
| Hermite | 21 | 3429.96 | 2.19e-7 | 0.99386–0.999995 |

To measure the full gradient error, at the same 4164 reference face centroids use

$$e_f=\|F_{p,f}-F_p^\star(X_f)\|_F,\qquad
 e_{\mathrm{mean}}=\frac1{4164}\sum_f e_f,\qquad e_{\max}=\max_f e_f.$$

These are absolute, dimensionless Frobenius errors. The mean weights faces
equally; it is not area-weighted and is not a relative error.

| Interpolant | Mean Fp error | Maximum Fp error |
|---|---:|---:|
| Q1 | 0.17501108 | 0.19452451 |
| Hermite | 0.00365963 | 0.00517635 |

Hermite has about 47.8 times smaller mean error and 37.6 times smaller maximum
error **for this field and these sample locations**. This supports its closer
approximation here, beyond determinant agreement alone. Lower final shell
energy is not an accuracy criterion: the two runs prescribe different Fp fields.

Checks cover affine reproduction of diag(1,.5,1), physical-gradient finite
differences on a translated non-unit box, and finite positive sampled
determinants. The analytic Fp formula was independently checked by finite
differences of the analytic position mapping. Both full solves reduce energy
and reach full nodal residual below 1e-5. No self-intersection or global-optimality
check is performed.

## Run and outputs

From the litepgo root after building its Python module:

```sh
uv run --no-sync python examples/fem/material/shell_interpolated_fp.py
uv run --no-sync python examples/fem/material/plot_shell_interpolated_fp.py
uv run --no-sync pytest tests/python/test_shell_interpolated_fp_example.py
```

`output/shell_interpolated_fp/` contains original.obj, q1_relaxed.obj,
hermite_relaxed.obj, the two field-target OBJs, per-basis NPZ data (including
sampled Fp and physical DOFs), summary.json and comparison.png.

<<< ../../../examples/fem/material/shell_interpolated_fp.py

<<< ../../../examples/fem/material/plot_shell_interpolated_fp.py

## C++ counterpart

The C++ executable runs both bases, checks affine reproduction and displacement
finite differences, then samples Fp and solves the same unconstrained problem.
It uses the same native response, basis-transform and shell-energy APIs as Python.

From a standalone litepgo checkout:

```sh
cmake --build build/release --target shell_interpolated_fp
build/release/examples/fem/material/shell_interpolated_fp
```

When built inside fit_micro, the executable instead lives at
`build/release/third_party/litepgo/examples/fem/material/shell_interpolated_fp`.
Outputs are in litepgo's `output/shell_interpolated_fp_cpp/`: original and relaxed
OBJs, `q1_Fp.csv`/`hermite_Fp.csv` (one face per row, column-major 3x3 entries),
and summary text files containing iterations, energy and full nodal residual.
The reference figure above is from the Python run; the C++ result is checked
against it numerically, allowing an arbitrary rigid pose for each free solve.

<<< ../../../examples/fem/material/shell_interpolated_fp.cpp

<<< ../../../examples/fem/material/plot_plastic_displacement.py
