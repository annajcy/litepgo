# FEM basics

These paired Python/C++ examples introduce the building blocks of finite elements.
They use generated reference-element data and write no files.

- [Coefficient sampling responses](sampling-response.md). Fixed points, variable
  coefficients, and value/gradient/Hessian sampling.
- [Spatial interpolation response](interpolated-response.md). Fixed coefficients,
  spatial Jacobians and exact weighted Hessians.
- [Tet P1 basis and field interpolation](tet-p1-basis.md). Evaluate shape functions,
  sample an affine target at vertices, and reconstruct it through a scalar Response.
- [Hermite interpolation](hermite-interpolation.md). Prepare value and derivative
  coefficients using sample descriptions; reconstruct a polynomial with the basis
  in 1D and with Responses in 2D/3D.
- [Quadrature integration](quadrature-integration.md). Integrate polynomials in
  1D/2D/3D and verify reference triangle/tetrahedron measures from weights.

- [Nonlinear response Hessians](response-hessian.md). Compose Triangle P1 sampling
  with exponentiation and verify the exact second-order energy chain rule.

Two Python-only examples write a material in NumPy and let native FEM assemble it;
see [materials written in Python](../../guide/fem/python-materials.md).

- [Volumetric material in NumPy](numpy-neohookean.md). Closed-form Neo-Hookean
  derivatives, Hermite hexes, both force responses and an equilibrium solve.
- [Shell material in NumPy](numpy-koiter-shell.md). Koiter StVK in packed forms,
  including the natural-form mixed derivative.

The [basis guide](../../guide/fem/basis.md) defines the supported formulations,
coefficient orders, and derivative layouts. Continue with
[quadrature](../../guide/fem/quadrature.md) for reference integration rules and
[fields and quadrature](../../guide/fem/fields.md) to combine shape functions
with coefficients and integrate local terms.
- [Free shell relaxation with prescribed Fp](shell-fp-free-relaxation.md).
  Koiter–StVK Schwarz P shell, open boundaries, fixed nonuniform plastic forms,
  and exported before/after geometry.
- [Uniform plastic contraction](shell-uniform-fp.md). The same Fp=diag(1,0.5,1)
  on every face, checked against the compatible affine zero-energy shape.
- [Shell plasticity from displacement interpolation](shell-interpolated-fp.md).
  Sample Q1/Hermite plastic displacement gradients at face centroids and compare free relaxation.
