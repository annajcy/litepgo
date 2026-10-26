# FEM with MeshEnergy

The `cpp/litepgo/fem/interpolation/` module contains generic field interpolation (`interpolation.hpp`), reference-geometry point data and evaluation (`rest_point.hpp`), and the Hex Hermite coefficient transform (`hex_hermite_dof_transform.hpp/.cpp`). Its tests mirror this layout in `tests/cpp/fem/interpolation/`.

## Notation and derivative layout

`x` is the global unknown vector. `x_local = map.gather(e, x)` preserves the map's existing matrix or vector shape; `dx` and `dx_local` are the corresponding increments. `e` indexes elements, `q` quadrature points, `a` basis functions, `v` vertices, and `c` component indices in array subscripts. `u_coeffs` names displacement coefficients when their physical meaning is known.

`x_position(X) = X + u(X)` is the deformed physical position, where `X` is the rest position and `u(X)` the displacement. The total deformation gradient is $F=\partial x_{\mathrm{position}}/\partial X$. Its multiplicative decomposition is $F=F_eF_p$: `Fp` is the plastic deformation gradient from the reference configuration to the plastic intermediate configuration, and `Fe` is the elastic deformation gradient from that intermediate configuration to the current configuration. All three are $3\times3$ matrices, with `Fe = F * inverse(Fp)`. `theta` is the model-specific parameter vector; Neo-Hookean uses `theta = [lambda, mu]` for the two Lamé parameters. In field formulas, $y$ denotes the output of `value()`, which may be `F`, `Fp`, or `theta`.

| Code name | Mathematical meaning |
| --- | --- |
| `xi` | Reference-element coordinates $\xi$ |
| `X` | Rest position $X$ at one point |
| `X_nodes` | Collection of rest nodal positions, in matrix or packed-vector form as specified by the interface |
| `rest_geometry`, `rest_geometries` | The interpolation map $\xi\mapsto X(\xi)$ for one rest element, or the collection of these maps |
| `dX_dxi` | $\partial X/\partial\xi$ |
| `dxi_dX` | $\partial\xi/\partial X$ |
| `det_dX_dxi` | $\det(\partial X/\partial\xi)$ |
| `N`, `dN_dxi` | Basis values $N_a$ and derivatives $\partial N_a/\partial\xi_d$ |
| `dN_dX` | Physical basis derivatives: `dN_dxi * dxi_dX` |
| `dvalue_dx_local` | $\partial\operatorname{vec}(y)/\partial x_{\mathrm{local}}$ at fixed geometry |
| `dF_dx_local` | The same derivative for the deformation gradient $F$ |
| `det_Fe`, `det_Fp`, `log_det_Fe` | $\det(F_e)$, $\det(F_p)$, $\log\det(F_e)$ |
| `inv_Fp`, `inv_Fe_T` | $F_p^{-1}$ and $F_e^{-T}$ |

`X_nodes` may be a `num_vertices`-by-3 matrix addressed as `X_nodes(v,c)`, or a vector of length `3*num_vertices` addressed as `X_nodes[3*v+c]`. Both store each vertex's three components in order; the declared interface type determines the representation. A name alone does not imply a matrix rank.

The denominator in `dvalue_dx_local` refers to the local unknowns of the object being differentiated:

| Context | Meaning of `x_local` |
| --- | --- |
| `MeshEnergy` callback | The matrix or vector returned by `map.gather(e, x)` |
| `Interpolation` / `AffineField` | The current object's own coefficient array, with scalar DOFs in that object's packing order |

Derivative matrices have output components in rows and scalar input DOFs in columns. Matrix outputs use column-major vectorization, independently of storage order: `vec(F)[c + 3*d] = F(c,d)`. Input packing stays field-specific: interpolation coefficients flatten by row, `flatten(x_local)[m*a + c] = coefficients(a,c)` for `m` components; `IdentityField` matrix DOFs flatten by column. Callbacks must account for any transformation between gathered unknowns and field coefficients when returning derivatives.

For example, the Hermite callback transforms physical derivative DOFs into reference coefficients. Writing the flattened reference coefficient vector as $c$ and the diagonal scaling matrix as $S=\operatorname{diag}(\texttt{dof\_scales})$, the fixed rest geometry gives

$$
c=S\,\operatorname{flatten}(x_{\mathrm{local}}),\qquad
g_{\mathrm{local}}=S^Tg_c,\qquad
H_{\mathrm{local}}=S^TH_cS.
$$

Here $g_c=\partial E_e/\partial c$ and $H_c=\partial^2 E_e/\partial c^2$. The output buffers named `dE_dx_local` and `d2E_dx_local2` temporarily contain these coefficient derivatives immediately after `integrate_volume`; the callback then applies the pullback above in place so its returned derivatives refer to the gathered physical `x_local`.

For `n` basis functions and `m` field components, `Interpolation` exposes:

| Method | Meaning | Shape |
| --- | --- | --- |
| `value(xi)` | $y(\xi)$ | $m$ |
| `dvalue_dxi(xi)` | $\partial y/\partial\xi$ | $m\times3$ |
| `dvalue_dx_local(xi)` | $\partial y/\partial x_{\mathrm{local}}$ | $m\times(nm)$ |
| `dgrad_xi_dx_local(xi)` | $\partial\operatorname{vec}(\partial y/\partial\xi)/\partial x_{\mathrm{local}}$ | $(3m)\times(nm)$ |

For the multiplicative material model, `psi` is elastic density per plastic volume and `W = det_Fp * psi` is density per original reference volume. The material-backed `LocalEnergy` methods return pointwise density and its DOF derivatives. `integrate_volume` computes $E_e=\int_{\Omega_e}W\,dV$, and `MeshEnergy` sums element energies into the global `E`. The discrete volume weight is `dV = weights[q] * point.det_dX_dxi`.

Material derivatives are `dW_dF`, `d2W_dF2`, and the corresponding `Fp` and `theta` names; other material arguments stay fixed. `dW_dF` and `dW_dFp` retain their $3\times3$ shape, with $\delta W=\sum_{ij}(\partial W/\partial F_{ij})\delta F_{ij}$; their Hessians are $9\times9$ matrices over column-major `vec(F)` or `vec(Fp)`. Elastic density derivatives use `dpsi_dFe` and `d2psi_dFe2`. Integrated element derivatives may be named `dE_dx_local` and `d2E_dx_local2`. Directional changes use `delta_`, such as `delta_Fe`; they are distinct from derivative operators such as `dFe_dF`.

Generic operations retain `value`, `gradient`, `hessian`, `evaluate`, `gather`, and `assemble`. This naming change preserves types, layouts, arithmetic, validation, and accumulation behavior. Existing finite-difference and layout tests cover the renamed APIs.

### C++ API migration

The old public names have been replaced; custom bases, fields, and materials must use the new names. No compatibility aliases are provided.

Update includes to the `fem/interpolation/` module as well:

| Previous include | Replacement |
| --- | --- |
| `litepgo/fem/geometry/interpolation.hpp` | `litepgo/fem/interpolation/interpolation.hpp` |
| `litepgo/fem/geometry/rest.hpp` | `litepgo/fem/interpolation/rest_point.hpp` |
| `litepgo/fem/geometry/hex_hermite_dof_transform.hpp` | `litepgo/fem/interpolation/hex_hermite_dof_transform.hpp` |

| Previous API | Replacement |
| --- | --- |
| `RestPointData::position` | `X` |
| `RestPointData::jacobian` | `dX_dxi` |
| `RestPointData::inverse_jacobian` | `dxi_dX` |
| `RestPointData::jacobian_determinant` | `det_dX_dxi` |
| `Basis::Sample::values`, `gradients` | `N`, `dN_dxi` (1D and 3D) |
| `Interpolation::gradient` | `dvalue_dxi` |
| `Interpolation::value_matrix` | `dvalue_dx_local` |
| `Interpolation::gradient_matrix` | `dgrad_xi_dx_local` |
| `AffineField::jacobian` and all overrides | `dvalue_dx_local` |
| Material `gradient_F`, `hessian_F` | `dW_dF`, `d2W_dF2` |
| Material `gradient_Fp`, `hessian_Fp` | `dW_dFp`, `d2W_dFp2` |
| Material `gradient_theta`, `hessian_theta` | `dW_dtheta`, `d2W_dtheta2` |
| `MultiplicativeKinematics::reference_gradient_F`, `reference_hessian_F` | `dW_dF`, `d2W_dF2` |
| `MultiplicativeKinematics::reference_gradient_Fp`, `reference_hessian_Fp` | `dW_dFp`, `d2W_dFp2` |

## Hex Q1

Two hexes of widths 1 and 2 share a face and its nodal displacement DOFs. Each `rest_geometries[e]` interpolates rest coordinates gathered for element `e`; `MeshEnergy` supplies that same element's `x_local` to the lambda. `integrate_volume<GaussLegendre3D<2>>` integrates that geometry with eight Gauss points. `MaterialLocalEnergyF<NeoHookean>` differentiates displacement DOFs; `Fp = I` and `theta = [2, 3]` stay fixed. The example evaluates energy, gradient, and Hessian, and checks both zero rest energy and uniform-strain energy over the total volume of 3.

From the repository root:

```sh
cmake --preset debug
cmake --build --preset debug --target hex_q1_mesh_energy
./build/debug/example/fem/hex_q1_mesh_energy
```

The target is excluded from the default build. This example evaluates energy and derivatives only; supply feasible step bounds before using a Newton solver.

The FD test uses the same setup from `hex_q1_mesh_energy.hpp` and checks the full gradient and Hessian at rest, under uniform stretch, and with nonuniform displacement:

```sh
cmake --build --preset debug --target litepgo_tests
./build/debug/tests/cpp/litepgo_tests --gtest_filter='HexQ1Example.*'
```

## Tet P1

Two tets share a face, with volumes `1/6` and `1/3`. They use `Interpolation<TetP1Basis>`, `DeformationGradientField<TetP1Basis>`, and `integrate_volume<TetCentroidQuadrature>` with the same `MeshEnergy` lambda pattern. Vertex order follows `(0,0,0), (1,0,0), (0,1,0), (0,0,1)` and must give a positive rest Jacobian.

The centroid rule has one point at `(1/4,1/4,1/4)` and weight `1/6`. With P1 geometry and displacement, `F` is constant per element, so the rule is exact for fixed `Fp` and `theta`. General spatially varying fields may require a higher-order rule. This example also evaluates energy and derivatives only.

```sh
cmake --build --preset debug --target tet_p1_mesh_energy litepgo_tests
./build/debug/example/fem/tet_p1_mesh_energy
./build/debug/tests/cpp/litepgo_tests --gtest_filter='TetP1Example.*'
```

The FD test in `tests/example/fem` shares the example setup and checks all 15 displacement DOFs at rest, under uniform stretch, and with nonuniform displacement.

## Hex Hermite

Two boxes of sizes `(1,2,3)` and `(2,2,3)` share a face. Geometry uses Hex Q1; displacement uses 64 Hermite coefficients per component. Each vertex has 24 physical DOFs: the three components of `u, u_X, u_Y, u_Z, u_XY, u_XZ, u_YZ, u_XYZ`.

The public function `hex_hermite_dof_transform(rest_geometry)` computes `S` from the fixed rest Q1 geometry, including the geometry and scale validation. Include `litepgo/fem/interpolation/hex_hermite_dof_transform.hpp` and compute it once per rest element:

```cpp
const auto S = litepgo::hex_hermite_dof_transform(rest_geometry);
const Eigen::VectorXd c = S * x_local.reshaped<Eigen::RowMajor>();
const Eigen::MatrixXd u_coeffs = c.reshaped<Eigen::RowMajor>(64, 3);
```

It returns a 192-by-192 `Eigen::DiagonalMatrix<double, Eigen::Dynamic>` storing only the 192 entries in `S.diagonal()`. For side lengths `hx`, `hy`, `hz`, the eight physical derivative slots scale by `1, hx, hy, hz, hx*hy, hx*hz, hy*hz, hx*hy*hz`, repeated for all three components at each vertex. Users supply the rest geometry and physical `x_local`; the function computes the transform. Since $S^T=S$, the Eigen expressions `S * g_c` and `S * H_c * S` apply the derivative pullback without an explicit transpose.

The example caches these transforms, converts physical derivatives to reference coefficients, integrates with `GaussLegendre3D<4>`, then returns `S^T g` and `S^T H S` through in-place diagonal scaling. Shared vertex DOFs enforce C1 continuity on conforming axis-aligned rest boxes in consistent Hex corner order.

Each rest element must be an axis-aligned rectangular box with positive dimensions; rotated, sheared, or warped rest hexes are rejected. The whole rest mesh need not form a box. This restriction applies only to the rest geometry: the deformed configuration `x_position(X) = X + u(X)` may rotate, stretch, shear, or bend, subject to the material's validity requirements. The DOF derivatives are with respect to fixed rest coordinates `X`, and their scales are computed once from the rest geometry. Supporting general rest hexes requires a different Hermite DOF mapping and compatible shared-face continuity conditions.

```sh
cmake --build --preset debug --target hex_hermite_mesh_energy litepgo_tests
./build/debug/example/fem/hex_hermite_mesh_energy
./build/debug/tests/cpp/litepgo_tests --gtest_filter='HexHermiteExample.*'
```

Tests in `tests/cpp/fem/interpolation` check the public transform's slot ordering, reproduction of physical mixed derivatives, and geometry/scale validation. Tests in `tests/example/fem` check affine fields, shared-face values and physical gradients, and energy derivatives with respect to physical DOFs. The example evaluates energy and derivatives only; feasible step bounds are not implemented. Nonlinear material integration is numerical.
