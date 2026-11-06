# FEM force responses

Force responses differentiate the positive displacement-gradient force with
respect to material or plastic coefficients at a fixed pose. They provide the
partial derivatives needed by fixed-pose fitting and
[equilibrium sensitivity](../solver/equilibrium.md).

## Fixed-pose force fitting

At prescribed displacement $u_0$, let

$$
f(p)=\nabla_u E(u_0,p),\qquad
r(p)=S^{-1}[f(p)-f_{\rm obs}],\qquad
\Phi(p)=\tfrac12\|r(p)\|^2.
$$

$S$ is a fixed observation scale. The Jacobian is
$J_r=S^{-1}B$, where $B=\partial(\nabla_uE)/\partial p$ holds
$u_0$ fixed. No displacement solve or equilibrium sensitivity appears here.

| Response template | Constructor fixed inputs | Parameter block |
| --- | --- | --- |
| `VolumetricFemForceMaterialResponse[M,B,Q]` | parameter map, displacement map, rest fields, displacement, Fp fields, batch size | Material parameters |
| `VolumetricFemForceFpResponse[M,B,Q]` | parameter map, displacement map, rest fields, displacement, theta fields, batch size | Nine column-packed Fp entries |
| `ShellFemForceMaterialResponse[M]` | parameter map, stencil map, rest data, displacement, thickness, batch size | Two plane-stress parameters |
| `ShellFemForceFpResponse[M]` | parameter map, stencil map, rest data, displacement, thickness, theta, batch size | Nine column-packed Fp entries |

Use `ConstantDofMap` for a shared block or `ElementDofMap` for independent
per-element blocks. Fixed geometry/pose arrays and maps must describe the same
element order. These responses snapshot the prescribed state; rebuild when it
changes. Fixed geometry and material spatial responses are sampled at construction and
are not retained. Active coefficients are explicit response inputs.

For example, after constructing the fixed geometry as in the
[solid guide](volume.md):

```python
Force = fem.VolumetricFemForceMaterialResponse[
    fem.NeoHookean, fem.TetP1Basis, fem.TetCentroidQuadrature]
force = Force(dof.ConstantDofMap(topology, 2), mapping, geometry, u0, plastic)
value, B = force.eval_value_and_jacobian(theta)
```

`value` has displacement-gradient force rows, and `B` has material-parameter
columns. The [batched force example](../../examples/energy/mesh-response.md)
provides a complete synthetic observation and fitting loop.

## Local force-response pullbacks

For a solid density $W$, let $P=\partial W/\partial F$ be first Piola stress.
Let $c_u$ and $c_\theta$ denote displacement and material coefficient vectors.
For an affine deformation field with coefficient derivative
$D_F=\partial\operatorname{vec}F/\partial c_u$ and parameter field derivative
$D_\theta=\partial\theta/\partial c_\theta$, a material-force local response is

$$
f_{\rm local}=D_F^T\operatorname{vec}P,\qquad
B_{\rm local}=D_F^T
\frac{\partial\operatorname{vec}P}{\partial\theta}D_\theta.
$$

These are densities before integration. `VolumetricElement3D` supplies the
quadrature/reference-volume weight once. Keep $F$, $F_p$, and $D_F$ fixed during
material-parameter differentiation. Plastic-force responses substitute
$\partial\operatorname{vec}P/\partial\operatorname{vec}F_p$ and the plastic
field's coefficient derivative. That mixed derivative includes the
$\det F_p$ factor and inverse-plastic chain rule; it is not simply a negative
elastic tangent. Reference/displacement basis transforms also pull force rows
back to physical DOFs.

The Field-dependent local material wrappers are removed. Use the pointwise material
functions and existing DOF pullback helpers for custom integration. Custom force maps use ordinary Response, with the active parameter vector as
input; there is no separate LocalResponse interface. See [the migration conventions](fields.md).
