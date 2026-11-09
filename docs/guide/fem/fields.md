# Spatial responses and coefficient maps

FEM uses the same `Response` interface as the rest of litepgo. A response differentiates its explicit input. There is no separate `Field` hierarchy and no implicit active coefficient block.

## Migrating existing code

| Previous API | Replacement |
| --- | --- |
| `InterpolatedField2D/3D[B](C)` | `InterpolatedResponse[B](C)`; derivatives now explicitly differentiate `xi` |
| `IdentityField2D/3D(c)` as fixed configuration | `AffineResponse(zeros((m,d)), c)`; coordinate derivatives are zero |
| Active coefficient derivatives of an interpolated field | The value, gradient and Hessian sampling factories at fixed points |
| `DeformationGradientField3D(u,X)` queried in space | `DeformationGradientResponse(u,X)`; its Jacobian differentiates `xi`, not `u` coefficients |
| Its active displacement-coefficient derivative | `deformation_gradient_sampling_response<B>(geometry, points).eval_jacobian(c)` |
| `SharedField3D` and concrete field template arguments | Ordinary Response ownership at the API boundary; FEM caches numerical samples |
| Field-dependent local material wrappers | Pointwise material functions and explicit DOF pullbacks, or a global FEM force response composed with coefficient sampling |

The removed interfaces have no compatibility aliases. In particular, do not mechanically
replace a coefficient derivative with `eval_jacobian(xi)`: those have different independent variables.

## Fixed coefficients, variable coordinates

For scalar basis functions \(N_a\), fixed coefficients \(C\in\mathbb R^{n_b\times m}\) define

\[
r(\xi)=C^T N(\xi),\qquad
J_r(\xi)=C^T D_\xi N(\xi).
\]

`InterpolatedResponse<B>(C)` in C++, or `fem.InterpolatedResponse[B](C)` in Python, owns a copy of `C`. Its input is a reference coordinate vector of length \(d\), including length one for 1D bases. Its output has length \(m\); its Jacobian has shape \((m,d)\). The weighted Hessian is the full symmetric matrix \(\sum_c w_c D_\xi^2 r_c\), with weights fixed during differentiation.

Rest positions are coefficients too: a geometry response uses \(m=3\) and \(C=X\), the element's rest-position coefficients. Geometry and displacement may use different bases. Coordinates outside the reference domain extrapolate; queries do not clamp.

A constant spatial response uses `AffineResponse(A, value)` with a zero \(m\times d\) matrix `A`. Its coordinate Jacobian is zero. To differentiate a material parameter directly, pass that parameter as the explicit input of the appropriate force response instead.

## Fixed coordinates, variable coefficients

With fixed points \(\xi_q\), the three [sampling factories](./sampling-responses) return `AffineResponse` objects:

\[
S_0 c = \{C^T N(\xi_q)\}_q,\quad
S_1 c = \{C^T D_\xi N(\xi_q)\}_q,\quad
S_2 c = \{C^T D_\xi^2 N(\xi_q)\}_q.
\]

Here \(c=\operatorname{vec}_{\rm row}(C)\). Outputs order point, spatial derivative slot, then component. Hessian samples store upper-triangle slots once, without mixed-entry scaling. The returned responses differentiate `c`, so their Jacobians are the constant sampling matrices and their own Hessians are zero. No quadrature weights or physical-coordinate transformation is included.

## Deformation and physical coordinates

For fixed displacement and geometry responses \(u(\xi),X(\xi)\),

\[
F(\xi)=I+D_\xi u(\xi)\,[D_\xi X(\xi)]^{-1}.
\]

`DeformationGradientResponse(displacement, geometry)` maps \(\xi\in\mathbb R^3\) to column-major `vec(F)`, with nine outputs. It owns its child responses, which must remain fixed, map three inputs to three outputs, and support Hessians. It requires a finite positive geometry determinant at the query point. Its exact spatial Jacobian uses

\[
\partial_k F=(\partial_k D_\xi u-(F-I)\partial_k D_\xi X)[D_\xi X]^{-1}.
\]

It does **not** provide Hessians: these would require third spatial derivatives. This map is useful for a prescribed plastic deformation. It does not differentiate displacement coefficients.

For active displacement coefficients, native FEM precomputes the dense matrix

\[
B_q=\bigl(D_\xi N(\xi_q)[D_\xi X(\xi_q)]^{-1}\bigr)^T\otimes I_3,
\qquad \operatorname{vec}(F_q)=\operatorname{vec}(I)+B_qc.
\]

`deformation_gradient_sampling_response<B>(geometry, points)` exposes this coefficient map as an `AffineResponse`. It takes reference coefficients, not physical DOFs.
Native FEM constructs this sampling Response at quadrature points and extracts its constant Jacobian into dense per-point blocks. It keeps those numerical blocks, not an extra Kinematics object. No response composition or sparse assembly is performed inside each constitutive evaluation.

## Fixed configuration is sampled once

`VolumetricFemEnergy<Basis,Quadrature>` accepts a batch material instance, a displacement map, and three lists of spatial responses for geometry, plastic matrices, and material parameters, in element order. Their input dimension is three and output dimensions are respectively three, nine and `material->num_parameters()`. C++ lists use `std::vector<std::shared_ptr<energy::Response>>`; Python accepts lists of response objects.

Construction caches integration weights, physical DOF transforms, `B_q`, and the material values at quadrature points. It retains no configuration responses. Force-response constructors likewise snapshot their fixed displacement and fixed material input. Rebuild the FEM object after changing configuration. Construction may call Python response hooks; numerical assembly subsequently uses the cached native data.

`VolumetricElement3D<Q>` and `BasisDofTransform<B>` accept a spatial geometry response and cache their required data. Nodal displacement coefficients are unchanged. Hex Hermite transforms physical DOFs using positive axis-aligned affine box lengths; this restriction has not been relaxed. `ElementLocator<B>` and `sample_boundary<B>` retain their P1/Q1 geometry restrictions: arbitrary spatial responses do not supply conservative geometric bounds. The locator snapshots its supported interpolated geometry.

## Integration and local material code

For \(w_q=\widehat w_q\det D_\xi X(\xi_q)\), native volumetric assembly computes

\[
E(c)=\sum_q w_q W(F_q,F_{p,q},\theta_q),\quad
\nabla_c E=\sum_q w_q B_q^T\operatorname{vec}(P_q),\quad
H_c=\sum_q w_q B_q^T A_qB_q.
\]

Physical DOFs are pulled back through `BasisDofTransform`. The force responses return the **positive energy gradient**, not its negative. Their parameter Jacobians replace \(A_q B_q\) with the material's mixed derivative with respect to `theta` or column-major `vec(Fp)`. `ConstantDofMap` shares a parameter block; `ElementDofMap` supplies one per element. Compose with coefficient sampling when the active parameter block itself is an affine function of other coefficients.

The old Field-dependent local material wrappers are removed. For custom integration, use the material's pointwise `eval_W`, stress and tangent functions, the existing DOF pullback helpers, and `VolumetricElement3D`'s C++ indexed accumulation callback. The `LocalEnergy2D/3D`, `LocalResponse2D/3D` and `DeformationGradientKinematics3D` APIs are removed. Custom energies and force responses use the ordinary Energy/Response interfaces, with active DOFs as inputs. Apply the complete integration weight once. Python exposes `element.points()` and `element.weights()` as independent arrays; the latter already includes the reference quadrature weight and geometry determinant.

The affine displacement formula makes the Hessian pullback above exact. A nonlinear map of active DOFs additionally needs its weighted second derivative and a suitable feasible-path rule; it cannot be substituted into this formula unchanged.

`max_feasible_step` returns a conservative bound in `[0,1]` for the entire closed step segment, not the largest feasible step. Volumetric FEM checks constitutive feasibility at quadrature points; it does not certify every spatial point of a curved element. Python `VolumetricElement3D.max_feasible_step(step)` accepts a callback `step(xi)` returning a conservative bound at that quadrature coordinate.

Ownership must remain acyclic. Where a response owns children, Python child-to-owner back-references must be weak references. See the [ownership convention](../energy/evaluation).

## Runtime assembly path

```text
physical displacement u
  -> gather element DOFs
  -> BasisDofTransform: reference coefficients c
  -> cached affine sampling blocks: vec(F_q) = vec(I) + B_q c
  -> pointwise material W, P, A
  -> weighted integration and coefficient derivatives
  -> physical-DOF pullback and global scatter
```

`InterpolatedResponse` and spatial `DeformationGradientResponse` can supply fixed
geometry, plastic deformation or material data at construction. They are not
created for each runtime displacement or used to differentiate active DOFs.
The sampling factory is the common construction path for both native FEM and
custom integration; see the [paired example](../../examples/fem/sampling-response.md).

Shells retain their triangle mid-edge geometric formulas. An internal function
takes the active stencil displacement explicitly and integrates its constant
face density. `ShellFemEnergy` owns fixed face data and assembles the result.
There is no public local-density inheritance layer. Its Hessian retains the
geometric second derivative; see [shell formulation](shell.md).
