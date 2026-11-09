# Coefficient sampling responses

These factories fix the Basis and reference query points, then return an ordinary
`energy.AffineResponse` whose input is the **coefficient vector**:

```python
value = fem.value_sampling_response[Basis](points, components=m)
gradient = fem.gradient_sampling_response[Basis](points, components=m)
hessian = fem.hessian_sampling_response[Basis](points, components=m)
```

C++ provides the same function templates in
`litepgo/fem/response/sampling_response.hpp`, returning `energy::AffineResponse`
by value; link `litepgo::fem`:

```cpp
auto value = fem::value_sampling_response<Basis>(points, m);
auto gradient = fem::gradient_sampling_response<Basis>(points, m);
auto hessian = fem::hessian_sampling_response<Basis>(points, m);
```

No additional response hierarchy is introduced. Python exposes the same seven
bases as [InterpolatedResponse](interpolated-response.md).

## Formulation and packing

Let $C\in\mathbb R^{k\times m}$ contain one row per basis coefficient and one
column per output component. The input is its row-major flattening:
$c_{am+i}=C_{ai}$. The interpolation is

$$
f_i(\xi;c)=\sum_{a=0}^{k-1}N_a(\xi)C_{ai}.
$$

The $Q$ fixed reference points are $\xi_q\in\mathbb R^d$.
Define $D_q\in\mathbb R^{k\times d}$ and
$S_q\in\mathbb R^{k\times s}$, where $s=d(d+1)/2$, by

$$
[D_q]_{aj}=\partial_{\xi_j}N_a(\xi_q),\qquad
[S_q]_{a\ell}=\partial_{\xi_r}\partial_{\xi_t}N_a(\xi_q).
$$

For $S_q$, slot $\ell$ enumerates $(r,t)$ in Basis upper-triangle order:
`(11)`, `(11,12,22)`, or `(11,12,13,22,23,33)`.
Mixed entries occur once and are **not multiplied by two**.

All three maps are linear, $R(c)=Ac$, with blocks stacked in query-point order:

| Factory | Block for point $q$ | Output length | Output order |
| --- | --- | --- | --- |
| `value_sampling_response` | $N(\xi_q)^T\otimes I_m$ | $Qm$ | point, component |
| `gradient_sampling_response` | $D_q^T\otimes I_m$ | $Qdm$ | point, coordinate direction, component |
| `hessian_sampling_response` | $S_q^T\otimes I_m$ | $Qsm$ | point, upper-triangle slot, component |

A gradient block is the column-major flattening of the $m\times d$ spatial
Jacobian. In Python its entire output may be reshaped as `(Q,d,m)`; Hessian
samples may be reshaped as `(Q,s,m)`, and values as `(Q,m)`.

## Derivatives of the sampling maps

For each factory, the returned Response obeys

$$
J_R=A,\qquad J_Rv=Av,\qquad J_R^Tw=A^Tw,\qquad H_R(c,w)=0.
$$

In particular, `hessian.evaluate(c).value` samples the spatial Hessian of $f$,
whereas `hessian.evaluate(c).hessian(w)` differentiates twice with respect to
**coefficients** and returns an empty sparse $(km,km)$ zero matrix.
Every sampling response reports `has_hessian == True`.

Returned Jacobians and Hessians are SciPy CSC matrices / C++ `SpMat`.
The factories evaluate the Basis once per point and retain the resulting matrix
snapshot. Exactly zero matrix entries are omitted at construction; later coefficient
changes do not change its structure. Results own their storage.

## Input and domain contracts

- `points` is a finite dense matrix of shape `(Q,d)`, including `(Q,1)` in 1D.
- `components` is an explicitly supplied positive integer.
- Empty point sets with shape `(0,d)` are allowed. Their responses have $km$
  inputs and zero outputs.
- Coefficient length is $km$; coefficients follow `Basis.coefficient_info` order.
  Hermite derivative coefficients remain derivatives with respect to reference coordinates.
- Dimensions and sparse entry counts must fit the library's integer sparse indices.
- Queries outside the reference domain extrapolate Basis polynomials without clamping.
  Non-finite Basis evaluations, including overflow, are rejected.
- The factories apply no quadrature weights, physical geometry transform or integration measure.

Gradient and Hessian here always mean **reference-coordinate derivatives**.
A physical gradient additionally requires the reference-to-physical geometry
Jacobian. Physical second derivatives for curved geometry require further chain-rule
terms; these factories do not perform that conversion.

## Relation to spatial interpolation

`InterpolatedResponse[Basis](C)` fixes coefficients and takes $\xi$ as input.
These sampling responses fix points and take $c$ as input. They describe the same
interpolation from different independent variables.

For nonlinear parameterizations, compose ordinary Responses explicitly.
For example, exponentiating sampled values computes $\exp(Ac)$; exponentiating
coefficients before sampling computes $A\exp(c)$. The two are generally different.

See the [paired sampling example](../../examples/fem/sampling-response.md).
The Field hierarchy has been removed. See [migration and contracts](fields.md) for the spatial-query and coefficient-map replacements.

## Deformation gradients in physical coordinates

`deformation_gradient_sampling_response<B>(geometry, points)` in C++, or
`fem.deformation_gradient_sampling_response[B](geometry, points)` in Python,
returns an ordinary `AffineResponse`. Python binds TetP1, HexQ1 and HexHermite.
For C++, include `litepgo/fem/response/deformation_gradient_sampling_response.hpp`.
`geometry` is a fixed spatial Response from reference coordinates to rest positions;
`points` is a finite `(Q,3)` array of reference coordinates, not physical nodes.
Empty point arrays are allowed. Geometry must have a finite positive Jacobian
 determinant at every sampled point.

For three-component displacement coefficients `C`, the input is
`c = C.ravel()` (C++ row-major flattening), with `3*B::num_basis_functions` entries.
The output has `9*Q` entries: point first, then column-major `vec(F)`:

$$
B_q = (D_\xi N(\xi_q)J_{X,q}^{-1})^T\otimes I_3,\qquad
\operatorname{vec}(F_q)=\operatorname{vec}(I)+B_qc.
$$

Its coefficient Jacobian stacks the fixed `B_q` matrices; weighted coefficient
Hessians are zero. Construction snapshots the numerical maps and retains neither
geometry nor points. Rebuild after changing geometry. No integration weights are
applied, and no physical-DOF conversion is hidden in this factory: for Hermite,
first use `BasisDofTransform` to obtain reference coefficients.

This differs from `DeformationGradientResponse(displacement, geometry)`, whose
input is a spatial coordinate and whose children remain fixed. Native FEM calls this factory at construction, then extracts its constant
Jacobian into dense per-point `B_q` blocks. Runtime material evaluation uses those
blocks directly. Spatial interpolation Responses do not enter that active-DOF path.

The [paired sampling example](../../examples/fem/sampling-response.md) includes
an anisotropically scaled geometry and checks the resulting physical gradient.
