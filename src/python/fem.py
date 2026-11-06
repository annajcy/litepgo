"""Finite elements, mirroring src/core/fem.

C++ class templates are Template objects: ``T[args]`` returns the bound
instantiation, with the C++ template arguments in order. Basis, quadrature and
material classes are the type arguments; integers stand for integer arguments.
Field type arguments of the energies are erased: they accept any Field3D.

InterpolatedResponse[Basis](coefficients) snapshots a finite (basis_count, components)
matrix and maps reference xi to coefficients.T @ Basis.eval_N(xi). All Response
derivatives are with respect to xi, not coefficients. Coordinates have shape (D,),
including (1,) for 1D bases. Weighted Hessians are exact full spatial Hessians.
All seven named bases are bound; polynomial queries extrapolate outside the domain.
See guide/fem/interpolated-response for shapes, packing and coefficient-map distinctions.

All named bases expose coefficient_info(i): point is a reference coordinate,
and derivative_order is a tuple of per-axis derivative orders. All zeros mean
a value sample.
LinearLagrangeBasis and CubicHermiteBasis take scalar 1D coordinates.

Quadrature types expose num_points, points() with shape (num_points, D),
and weights() with shape (num_points,). Weights include reference-domain measure;
physical geometry is applied by the integrator. GaussLegendre1D/2D/3D[n]
support n=2 or 4 points per axis and return independent NumPy array copies.

    Energy = fem.VolumetricFemEnergy[fem.NeoHookean, fem.HexQ1Basis, fem.GaussLegendre3D[2]]
    energy = Energy(displacement_map, rest_position_fields, Fp, theta)

NeoHookean.eval_d2W_dF_dtheta(F, Fp, theta) returns a (9, 2) analytic stress
parameter Jacobian at fixed F and Fp. Rows flatten stress in column-major order;
columns follow theta=[lambda, mu]. It includes the reference-volume det(Fp)
factor and does not differentiate equilibrium or plastic evolution.

NeoHookean.eval_d2W_dF_dFp(F, Fp, theta) returns the analytic (9, 9)
stress/plastic Jacobian at fixed F and theta, with both axes column-major.
`VolumetricFpForceLocalResponse3D[Material](F, Fp, theta)` differentiates force
with respect to the Fp field's affine coefficients. VolumetricFemForceFpResponse
[NeoHookean, TetP1Basis, TetCentroidQuadrature] takes the same arguments as the
material mesh response but with fixed theta fields replacing Fp fields. Its
parameter_map has nine DOFs per block, and eval accepts column-major vec(Fp).
Both ForceResponse families have the same basis/quadrature instantiations as
VolumetricFemEnergy: TetP1/TetCentroid, and HexQ1 or HexHermite with
GaussLegendre3D[2] or GaussLegendre3D[4].
Displacements and output forces use BasisDofTransform's physical DOF convention;
Hermite uses 24 DOFs per vertex on positive axis-aligned affine reference boxes.

`ShellFemEnergy[KoiterStVK](map, rest, thickness, theta, Fp=[])` accepts optional
fixed per-face 3x3 plastic matrices. Only natural forms change; original positions,
area weights and thickness stay fixed. Empty Fp preserves the supplied rest forms.
It snapshots all inputs and differentiates displacement, unlike the Fp response.

LocalResponse2D/3D mirror LocalEnergy with num_inputs, num_outputs, eval_value,
has_jacobian and eval_jacobian. VolumetricMaterialForceLocalResponse3D[Material]
borrows (F, Fp, theta), keeps Python fields alive, and returns the positive
F-coefficient force density and its theta-coefficient Jacobian. F and Fp stay
fixed. accumulate(xi, weight, jacobian=False) returns weighted outputs; geometry
is applied only by VolumetricElement3D.eval(local_response, jacobian=False).
The integrator uses accumulate_into and validates final results;
this strided C++ entry point is not exposed directly in Python, as for LocalEnergy.

VolumetricFemForceMaterialResponse[NeoHookean, TetP1Basis, TetCentroidQuadrature]
takes (parameter_map, displacement_map, rest_position_fields, displacement, Fp,
batch_size=256). It copies displacement and owns/shares fields, which must remain
unchanged. ConstantDofMap shares [lambda, mu]; ElementDofMap assigns a pair per
element. eval(theta, need_jacobian=True) returns global positive nodal force and
a sparse parameter Jacobian. Constitutive evaluation and integration run through
the local response API, with batched TBB assembly and no equilibrium solve.

ShellFemForceMaterialResponse[KoiterStVK] takes (parameter_map, displacement_map,
rest_data, displacement, thickness, batch_size=256). It snapshots fixed inputs,
caches pose kinematics and returns positive displacement-gradient force and its
theta Jacobian. The output map is StencilDofMap; parameters use ConstantDofMap or
ElementDofMap with [lambda_ps, mu] per block. Thickness is fixed.
`ShellMaterialForceLocalResponse2D[KoiterStVK](rest, u_local, thickness, theta)`
borrows a Field2D for theta and returns density per reference area; accumulate
applies a caller-supplied weight. Python keeps theta alive.

ShellFemForceFpResponse[KoiterStVK] takes (parameter_map, displacement_map,
rest_data, displacement, thickness, theta, batch_size=256). Fp is a column-major
nine-vector shared globally or constant per face. The entire reference stencil
is transformed by Fp; current positions, original area weights and thickness stay
fixed. `ShellFpForceLocalResponse2D[KoiterStVK](rest, u_local, thickness, Fp, theta)`
borrows a nine-component Field2D for Fp. Its Jacobian follows field coefficients;
IdentityField2D gives the full nine-column vec(Fp) Jacobian. Both require det(Fp)>0.

Only the instantiations compiled into the extension exist; indexing any other
raises TypeError listing them. Every instantiation is an ordinary class, and
``isinstance(x, Template)`` tests membership in any of them.
"""

from litepgo._core import fem as _native


class Template:
    """A C++ class template whose bound instantiations are looked up by argument tuple."""

    def __init__(self, name, instances):
        self.__name__ = name
        self._instances = dict(instances)

    def __getitem__(self, args):
        key = args if isinstance(args, tuple) else (args,)
        try:
            return self._instances[key]
        except KeyError:
            raise TypeError(
                f"{self.__name__}[{_format(key)}] is not instantiated; available: "
                + "; ".join(f"{self.__name__}[{_format(k)}]" for k in self._instances)
            ) from None

    @property
    def instances(self):
        """A copy of {argument tuple: class} for every bound instantiation."""
        return dict(self._instances)

    def __instancecheck__(self, obj):
        return isinstance(obj, tuple(self._instances.values()))

    def __subclasscheck__(self, cls):
        return issubclass(cls, tuple(self._instances.values()))

    def __repr__(self):
        return f"<template litepgo.fem.{self.__name__}>"


def _format(key):
    return ", ".join(getattr(arg, "__name__", repr(arg)) for arg in key)


_templates = {name: Template(name, instances) for name, instances in _native._templates.items()}

CoefficientInfo1D = _native.CoefficientInfo1D
CoefficientInfo2D = _native.CoefficientInfo2D
CoefficientInfo3D = _native.CoefficientInfo3D
LinearLagrangeBasis = _native.LinearLagrangeBasis
CubicHermiteBasis = _native.CubicHermiteBasis
TriangleP1Basis = _native.TriangleP1Basis
QuadHermiteBasis = _native.QuadHermiteBasis
TetP1Basis = _native.TetP1Basis
HexQ1Basis = _native.HexQ1Basis
HexHermiteBasis = _native.HexHermiteBasis

TriangleCentroidQuadrature = _native.TriangleCentroidQuadrature
TetCentroidQuadrature = _native.TetCentroidQuadrature
GaussLegendre1D = _templates["GaussLegendre1D"]
GaussLegendre2D = _templates["GaussLegendre2D"]
GaussLegendre3D = _templates["GaussLegendre3D"]

build_point_displacement_map = _templates["build_point_displacement_map"]

sample_boundary = _templates["sample_boundary"]
SampledBoundary = _native.SampledBoundary

ElementLocator = _templates["ElementLocator"]
PointEmbedding = _native.PointEmbedding

Field2D = _native.Field2D
Field3D = _native.Field3D
IdentityField2D = _native.IdentityField2D
IdentityField3D = _native.IdentityField3D
InterpolatedResponse = _templates["InterpolatedResponse"]
InterpolatedField2D = _templates["InterpolatedField2D"]
InterpolatedField3D = _templates["InterpolatedField3D"]
DeformationGradientField3D = _native.DeformationGradientField3D

NeoHookean = _native.NeoHookean
KoiterStVK = _native.KoiterStVK

TriangleMidEdgeRestData = _native.TriangleMidEdgeRestData
TriangleMidEdgeKinematics = _native.TriangleMidEdgeKinematics
eval_triangle_mid_edge_kinematics = _native.eval_triangle_mid_edge_kinematics
eval_triangle_mid_edge_feasible_step = _native.eval_triangle_mid_edge_feasible_step
eval_positive_det_feasible_step = _native.eval_positive_det_feasible_step
eval_gradient_wrt_dofs = _native.eval_gradient_wrt_dofs
eval_hessian_wrt_dofs = _native.eval_hessian_wrt_dofs

LocalEnergy2D = _native.LocalEnergy2D
LocalEnergy3D = _native.LocalEnergy3D
LocalResponse2D = _native.LocalResponse2D
LocalResponse3D = _native.LocalResponse3D
VolumetricFpForceLocalResponse3D = _templates["VolumetricFpForceLocalResponse3D"]
VolumetricMaterialForceLocalResponse3D = _templates["VolumetricMaterialForceLocalResponse3D"]
VolumetricMaterialLocalEnergyF3D = _templates["VolumetricMaterialLocalEnergyF3D"]
ShellMaterialLocalEnergy2D = _templates["ShellMaterialLocalEnergy2D"]
VolumetricElement3D = _templates["VolumetricElement3D"]
BasisDofTransform = _templates["BasisDofTransform"]

VolumetricFemEnergy = _templates["VolumetricFemEnergy"]
VolumetricFemForceFpResponse = _templates["VolumetricFemForceFpResponse"]
VolumetricFemForceMaterialResponse = _templates["VolumetricFemForceMaterialResponse"]
ShellFemEnergy = _templates["ShellFemEnergy"]
ShellMaterialForceLocalResponse2D = _templates["ShellMaterialForceLocalResponse2D"]
ShellFemForceMaterialResponse = _templates["ShellFemForceMaterialResponse"]
ShellFpForceLocalResponse2D = _templates["ShellFpForceLocalResponse2D"]
ShellFemForceFpResponse = _templates["ShellFemForceFpResponse"]

__all__ = [
    "sample_boundary",
    "SampledBoundary",
    "build_point_displacement_map",
    "ElementLocator",
    "PointEmbedding",
    "Template",
    "CoefficientInfo1D",
    "CoefficientInfo2D",
    "CoefficientInfo3D",
    "LinearLagrangeBasis",
    "CubicHermiteBasis",
    "TriangleP1Basis",
    "QuadHermiteBasis",
    "TetP1Basis",
    "HexQ1Basis",
    "HexHermiteBasis",
    "TriangleCentroidQuadrature",
    "TetCentroidQuadrature",
    "GaussLegendre1D",
    "GaussLegendre2D",
    "GaussLegendre3D",
    "Field2D",
    "Field3D",
    "IdentityField2D",
    "IdentityField3D",
    "InterpolatedResponse",
    "InterpolatedField2D",
    "InterpolatedField3D",
    "DeformationGradientField3D",
    "NeoHookean",
    "KoiterStVK",
    "TriangleMidEdgeRestData",
    "TriangleMidEdgeKinematics",
    "eval_triangle_mid_edge_kinematics",
    "eval_triangle_mid_edge_feasible_step",
    "eval_positive_det_feasible_step",
    "eval_gradient_wrt_dofs",
    "eval_hessian_wrt_dofs",
    "LocalEnergy2D",
    "LocalEnergy3D",
    "LocalResponse2D",
    "LocalResponse3D",
    "VolumetricMaterialForceLocalResponse3D",
    "VolumetricFpForceLocalResponse3D",
    "VolumetricMaterialLocalEnergyF3D",
    "ShellMaterialLocalEnergy2D",
    "VolumetricElement3D",
    "BasisDofTransform",
    "VolumetricFemEnergy",
    "VolumetricFemForceMaterialResponse",
    "VolumetricFemForceFpResponse",
    "ShellFemEnergy",
    "ShellMaterialForceLocalResponse2D",
    "ShellFemForceMaterialResponse",
    "ShellFpForceLocalResponse2D",
    "ShellFemForceFpResponse",
]

LogDiagonalFpResponse = _native.LogDiagonalFpResponse
__all__ += ["LogDiagonalFpResponse"]
