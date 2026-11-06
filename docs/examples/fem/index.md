# FEM basics

These paired Python/C++ examples introduce the building blocks of finite elements.
They use generated reference-element data and write no files.

- [Tet P1 basis and field interpolation](tet-p1-basis.md). Evaluate shape functions,
  sample an affine target at vertices, and reconstruct it through a scalar Field.
- [Hermite interpolation](hermite-interpolation.md). Prepare value and derivative
  coefficients using sample descriptions; reconstruct a polynomial with the basis
  in 1D and with Fields in 2D/3D.
- [Quadrature integration](quadrature-integration.md). Integrate polynomials in
  1D/2D/3D and verify reference triangle/tetrahedron measures from weights.

- [Nonlinear response Hessians](response-hessian.md). Compose Triangle P1 sampling
  with exponentiation and verify the exact second-order energy chain rule.

The [basis guide](../../guide/fem/basis.md) defines the supported formulations,
coefficient orders, and derivative layouts. Continue with
[quadrature](../../guide/fem/quadrature.md) for reference integration rules and
[fields and quadrature](../../guide/fem/fields.md) to combine shape functions
with coefficients and integrate local terms.
