# Volumetric elasticity and prescribed plasticity

A volumetric energy maps displacement DOFs to a scalar elastic energy. Geometry,
plastic state, and material fields are fixed during each energy's displacement
solve. This chapter defines the implemented model and builds one complete element.

## Solid evaluation pipeline

The high-level energy gathers physical displacement DOFs and transforms them
before applying cached deformation-gradient matrices. Material derivatives travel back through the same
kinematics and transform; reference volume is applied once during integration.

```mermaid
flowchart TB
  accTitle: Volumetric energy and derivative pipeline
  accDescr: Physical displacement DOFs become reference coefficients and a displacement field. Rest geometry converts its derivative to F. Material evaluation uses F, prescribed Fp and theta; quadrature and pullbacks return physical local derivatives for assembly.
  dofs["Physical element displacement DOFs"] --> transform["BasisDofTransform"]
  transform --> field["Displacement coefficients"]
  field --> F["vec(F) = vec(I) + B_q c"]
  geometry["Rest geometry response (sampled at construction)"] --> F
  F --> material["Material density and derivatives"]
  fixed["Cached Fp and theta samples"] --> material
  material --> integrate["Rest-volume quadrature"]
  integrate --> pullback["Physical DOF pullback and assembly"]
```

The arrows show forward evaluation order; gradient and Hessian pullbacks apply
the chain rule in reverse. The same local ingredients also support fixed-pose
material/plastic force responses.

## Kinematics and Neo-Hookean density

Let $F=I+\nabla_Xu$ be the total deformation gradient and $F_p$ a prescribed
plastic gradient. Multiplicative kinematics uses

$$
F=F_eF_p,\qquad F_e=FF_p^{-1},\qquad J_e=\det F_e.
$$

`NeoHookean` uses material parameters $\theta=(\lambda,\mu)$ and density per
original rest volume

$$
\psi(F_e)=\frac\mu2(\|F_e\|_F^2-3)-\mu\log J_e
+\frac\lambda2(\log J_e)^2,
$$

$$
W(F,F_p,\theta)=\det(F_p)\,\psi(F_e).
$$

The admissible domain has $\det F>0$, $\det F_p>0$, $\lambda\ge0$,
and $\mu>0$, with finite representable evaluations. For $F_p=I$, the rest
state $F=I$ has zero energy and stress. The first Piola stress is

$$
P=\frac{\partial W}{\partial F}
=\det(F_p)\left[\mu F_e+(\lambda\log J_e-\mu)F_e^{-T}\right]F_p^{-T}.
$$

`eval_dW_dF` returns the $3\times3$ stress. `eval_d2W_dF2` returns its
$9\times9$ derivative using column-major vectorization. The gradient of
integrated energy is the positive nodal force used by the examples.

## Complete one-tet energy

```python
import numpy as np
from litepgo import dof, energy, fem, mesh

X = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
topology = mesh.TetTopology(4, np.array([[0, 1, 2, 3]], dtype=np.int32))
mapping = dof.VertexDofMap(topology, 3)
XField = fem.InterpolatedResponse[fem.TetP1Basis]
geometry = [XField(mapping.gather(0, X.ravel()))]
plastic = [energy.AffineResponse(np.zeros((9, 3)), np.eye(3).ravel(order="F"))]
material = [energy.AffineResponse(np.zeros((2, 3)), np.array([2., 3.]))]
Energy = fem.VolumetricFemEnergy[fem.NeoHookean, fem.TetP1Basis,
                                  fem.TetCentroidQuadrature]
elastic = Energy(mapping, geometry, plastic, material)
u = np.zeros(mapping.num_dofs)
value, gradient, hessian = elastic.eval(u, gradient=True,
                                       hessian=energy.HessianKind.Exact)
assert abs(value) < 1e-12
assert np.linalg.norm(gradient) < 1e-12
assert hessian.shape == (12, 12)
```

For multiple elements, pass one geometry, plastic and material spatial response per element,
in the same order as the DOF map. Construction snapshots the needed values and derivatives;
no configuration response is retained. Rebuild after changing fixed configuration. Geometry
and displacement may use different bases.

## Basis and quadrature choices

| Displacement | DOFs per vertex | Typical geometry | Bound quadrature |
| --- | --- | --- | --- |
| Tet P1 | 3 | Tet P1 | Tet centroid |
| Hex Q1 | 3 | Hex Q1 | Gauss–Legendre 2 or 4 per axis |
| Hex Hermite | 24 | Axis-aligned affine Hex Q1 boxes | Gauss–Legendre 2 or 4 per axis |

C++ and Python both have three template arguments: Material, DisplacementBasis and Quadrature.
Python `.instances` lists compiled combinations. See [Hex Hermite](hermite.md) before constructing a 24-DOF map.

The global energy is

$$
E(u)=\sum_{e,q}w_q\det J_{X,e}(\xi_q)
 W\!\left(I+u_{e,\xi}J_{X,e}^{-1},F_{p,e},\theta_e\right).
$$

`VolumetricFemEnergy` caches geometry weights and dense kinematics, evaluates the material,
and assembles through `MeshEnergy`. For custom integration, use pointwise material functions
and the indexed accumulation callback of `VolumetricElement3D`; see
[spatial responses and coefficient maps](fields.md#integration-and-local-material-code) for
formulas, packing and the replacement for the removed Field-dependent local wrappers.

## Plastic fields and parameter-force derivatives

An affine response with a zero matrix describes a constant $F_p$; a
`DeformationGradientResponse(u_p, X)` describes
$F_p=I+\nabla_Xu_p$, where $u_p$ is a prescribed compatible displacement.
The latter includes shear induced by a spatially varying rotation. It is not
merely a rotation matrix field. If the displacement space contains $u_p$,
$u=u_p$ is a stress-free state when boundary conditions permit it.

`VolumetricFemForceMaterialResponse` varies material coefficients at a fixed
pose. `VolumetricFemForceFpResponse` varies column-packed plastic coefficients at
a fixed pose. They expose $f=\partial E/\partial u$ and the mixed derivatives
$\partial f/\partial\theta$ or $\partial f/\partial\operatorname{vec}F_p$.
They do not solve equilibrium. See [inverse problems](../solver/inverse.md) for total
derivatives through a solve and [plasticity examples](../../examples/plasticity/index.md)
for prescribed-field equilibria.

## Feasibility and scope

The solid energy bounds steps using the same quadrature points as evaluation.
`eval_positive_det_feasible_step(A, dA)` bounds the first zero of
$\det(A+\alpha dA)$, including interior/repeated roots, and leaves a safety
margin. Tet P1 has constant $F$ per element; Hex Q1/Hermite quadrature checks
are not a certificate of injectivity everywhere inside the element. Positive
determinants also do not prevent floating-point overflow at extreme scales.

These are elastic solves with a supplied plastic state. Yield surfaces,
plastic-state evolution, dynamic time integration, and contact are not implicitly
added. Compose contact or loads explicitly in a total potential.
