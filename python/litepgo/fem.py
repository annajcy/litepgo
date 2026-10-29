"""Finite elements, mirroring cpp/litepgo/fem.

C++ class templates are Template objects: ``T[args]`` returns the bound
instantiation, with the C++ template arguments in order. Basis, quadrature and
material classes are the type arguments; integers stand for integer arguments.
Field type arguments of the energies are erased: they accept any Field3D.

    Energy = fem.VolumetricFemEnergy[fem.NeoHookean, fem.HexQ1Basis, fem.GaussLegendre3D[2]]
    energy = Energy(displacement_map, rest_position_fields, Fp, theta)

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

TriangleP1Basis = _native.TriangleP1Basis
QuadHermiteBasis = _native.QuadHermiteBasis
TetP1Basis = _native.TetP1Basis
HexQ1Basis = _native.HexQ1Basis
HexHermiteBasis = _native.HexHermiteBasis

TriangleCentroidQuadrature = _native.TriangleCentroidQuadrature
TetCentroidQuadrature = _native.TetCentroidQuadrature
GaussLegendre2D = _templates["GaussLegendre2D"]
GaussLegendre3D = _templates["GaussLegendre3D"]

Field2D = _native.Field2D
Field3D = _native.Field3D
IdentityField2D = _native.IdentityField2D
IdentityField3D = _native.IdentityField3D
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
VolumetricMaterialLocalEnergy3D = _templates["VolumetricMaterialLocalEnergy3D"]
VolumetricMaterialLocalEnergyF3D = _templates["VolumetricMaterialLocalEnergyF3D"]
VolumetricMaterialLocalEnergyFp3D = _templates["VolumetricMaterialLocalEnergyFp3D"]
VolumetricMaterialLocalEnergyTheta3D = _templates["VolumetricMaterialLocalEnergyTheta3D"]
ShellMaterialLocalEnergy2D = _templates["ShellMaterialLocalEnergy2D"]
VolumetricElement3D = _templates["VolumetricElement3D"]
BasisDofTransform = _templates["BasisDofTransform"]

VolumetricFemEnergy = _templates["VolumetricFemEnergy"]
ShellFemEnergy = _templates["ShellFemEnergy"]

__all__ = [
    "Template",
    "TriangleP1Basis",
    "QuadHermiteBasis",
    "TetP1Basis",
    "HexQ1Basis",
    "HexHermiteBasis",
    "TriangleCentroidQuadrature",
    "TetCentroidQuadrature",
    "GaussLegendre2D",
    "GaussLegendre3D",
    "Field2D",
    "Field3D",
    "IdentityField2D",
    "IdentityField3D",
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
    "VolumetricMaterialLocalEnergy3D",
    "VolumetricMaterialLocalEnergyF3D",
    "VolumetricMaterialLocalEnergyFp3D",
    "VolumetricMaterialLocalEnergyTheta3D",
    "ShellMaterialLocalEnergy2D",
    "VolumetricElement3D",
    "BasisDofTransform",
    "VolumetricFemEnergy",
    "ShellFemEnergy",
]
