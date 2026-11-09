"""Finite elements using Response for spatial queries and coefficient sampling.

InterpolatedResponse[Basis](coefficients) maps xi to coefficients.T @ N(xi).
value_sampling_response, gradient_sampling_response and hessian_sampling_response
map row-major coefficients to samples at fixed reference points. Their derivatives
are with respect to coefficients; spatial derivative samples remain linear maps.

VolumetricFemEnergy[Basis, Quadrature] samples geometry, Fp and theta
spatial Responses at construction. Rebuild after changing fixed configuration.
VolumetricFemForceMaterialResponse and VolumetricFemForceFpResponse similarly
snapshot fixed displacement and configuration, differentiating active parameters.
deformation_gradient_sampling_response[Basis](geometry, points) returns an
AffineResponse from row-major three-component displacement coefficients to
point-major, column-major vec(F). It snapshots geometry at (Q,3) reference points;
no physical DOF transform or quadrature weights are included.

DeformationGradientResponse(displacement, geometry) maps xi to column-major
vec(I + Du @ inverse(DX)). Its children map R3 to R3, remain fixed and support
Hessians. It provides exact spatial Jacobians but no Hessians of its own.

Basis classes expose scalar shape functions N, dN_dxi, d2N_dxi2 and
coefficient_info(i). A coefficient sample specifies its reference coordinate
and per-axis derivative order. Quadrature points have shape (Q,D), including
(Q,1) in 1D; weights include reference-domain measure. Physical geometry is
applied once during integration.

C++ class/function templates are indexed as T[arguments]. Only compiled
instantiations are available; instances lists those choices. Volumetric FEM
selects displacement Basis and Quadrature in both C++ and Python. Material
instances are the first constructor argument.
TetP1 uses 3 physical displacement DOFs per vertex. HexHermite uses 24 on
positive axis-aligned affine boxes; BasisDofTransform converts them to reference
coefficients and pulls derivatives back. Geometry may use a different basis.

Force responses return the positive displacement gradient at fixed pose.
ConstantDofMap shares one parameter block; ElementDofMap assigns one per element.
Volumetric material blocks follow the material parameter order; plastic blocks
are column-major vec(Fp). These partial derivatives do not solve equilibrium.

ShellFemEnergy and shell force responses snapshot rest stencils, thickness and
fixed parameters. ShellFemEnergy accepts optional per-face plastic matrices;
these change natural forms, preserving original area weights and thickness.
Shell plastic force responses differentiate nine column-major Fp entries.

Python materials subclass PyVolumetricMaterial or PyShellMaterial. Pass an instance
to FEM, e.g. VolumetricFemEnergy[Basis, Quadrature](MyMaterial(), ...). Native
pointwise formulas use PointwiseVolumetricAdapter[NeoHookean]() or
PointwiseShellAdapter[KoiterStVK](). Material eval receives one batch per call.

Native volumetric FEM constructs its fixed maps through deformation-gradient
sampling. Custom integration uses ordinary Energy/Response, pointwise material
derivatives, and VolumetricElement3D.points()/weights(); weights already include
the geometry determinant. There is no separate local-density interface.
See the FEM guide for formulas, shapes, packing and feasibility conventions.
"""

from litepgo._core import fem as _native


class PyVolumetricMaterial(_native.PyVolumetricMaterial):
    """Base class for volumetric materials written in Python (NumPy, Numba, Torch, ...).

    Subclass it and pass an instance as the first argument to VolumetricFemEnergy,
    VolumetricFemForceMaterialResponse or VolumetricFemForceFpResponse. Define
    num_parameters on the class and eval on the instance. Call super().__init__()
    when providing a constructor. FEM objects retain the material instance.

    Required:

    - ``num_parameters``: int, the width k of theta.
    - ``eval(F, Fp, theta, *, outputs) -> dict``. F and Fp have shape (N,3,3)
      and theta (N,k), CPU float64 NumPy arrays whose rows are independent
      points. ``outputs`` is a tuple of the names below; return a dict holding
      at least those entries (extra entries are ignored). Arrays are converted
      with numpy.ascontiguousarray(..., float64).

    ==================  ===============  =============================================
    name                shape            entry
    ==================  ===============  =============================================
    ``W``               (N,)             energy density per original reference volume
    ``dW_dF``           (N,3,3)          [n,i,j] = dW/dF_ij
    ``d2W_dF2``         (N,3,3,3,3)      [n,i,j,k,l] = d2W/dF_ij dF_kl
    ``d2W_dF_dtheta``   (N,3,3,k)        [n,i,j,a] = d2W/dF_ij dtheta_a
    ``d2W_dF_dFp``      (N,3,3,3,3)      [n,i,j,k,l] = d2W/dF_ij dFp_kl
    ==================  ===============  =============================================

    Materials must support every listed output, including mixed derivatives.
    Derivatives hold the other inputs fixed. Energies request ("W",),
    ("W","dW_dF") or ("W","dW_dF","d2W_dF2"); material force responses request
    ("dW_dF",) or ("dW_dF","d2W_dF_dtheta"); Fp force responses ("dW_dF",) or
    ("dW_dF","d2W_dF_dFp").

    Optional:

    - ``eval_feasible_step(F, Fp, theta, dF) -> (N,)``: a conservative bound in
      [0,1] keeping each point feasible on F + t dF, t in [0, bound]. The inherited
      default keeps det(F) > 0 and can be called directly or with super().

    Non-finite returned values reject the evaluation like a native domain error
    (a Newton trial step is then shortened); exceptions raised by eval propagate.
    Each batch enters Python with the GIL held. Native Python extensions can release
    it, so callbacks must protect shared mutable state against overlapping calls.
    """

    num_parameters = None

    def eval(self, F, Fp, theta, *, outputs):
        raise NotImplementedError("PyVolumetricMaterial subclasses must implement eval")


class PyShellMaterial(_native.PyShellMaterial):
    """Base class for triangle mid-edge shell materials written in Python.

    Subclass it and pass an instance as the first argument to ShellFemEnergy,
    ShellFemForceMaterialResponse or ShellFemForceFpResponse. Define num_parameters
    on the class and call super().__init__() in custom constructors.

    Required:

    - ``num_parameters``: int, the width k of theta.
    - ``eval(ab, ab_bar, thickness, theta, *, outputs) -> dict``. ab and ab_bar
      hold packed current and natural forms (a11,a12,a22,b11,b12,b22) with
      shape (N,6); thickness has shape (N,) and theta (N,k). Mixed entries are
      independent, unscaled variables, so differentiating W with respect to the
      packed vector gives the expected derivatives.

    ======================  =========  ===========================================
    name                    shape      entry
    ======================  =========  ===========================================
    ``W``                   (N,)       energy density per original reference area
    ``dW_dab``              (N,6)      [n,r] = dW/dab_r
    ``d2W_dab2``            (N,6,6)    [n,r,s] = d2W/dab_r dab_s
    ``d2W_dab_dtheta``      (N,6,k)    [n,r,a] = d2W/dab_r dtheta_a
    ``d2W_dab_dabarbbar``   (N,6,6)    [n,r,s] = d2W/dab_r dab_bar_s
    ======================  =========  ===========================================

    Materials must support every listed output, including mixed derivatives.
    Energies request ("W",), ("W","dW_dab") or ("W","dW_dab","d2W_dab2");
    force responses request "dW_dab" with "d2W_dab_dtheta" or
    "d2W_dab_dabarbbar" for Jacobians. Plastic Fp responses chain the natural
    forms to Fp natively. Line searches use only the geometric shell bound, so
    the material must be finite wherever the current forms stay positive definite.
    """

    num_parameters = None

    def eval(self, ab, ab_bar, thickness, theta, *, outputs):
        raise NotImplementedError("PyShellMaterial subclasses must implement eval")


class Template:
    """A C++ template whose bound classes or factory functions are looked up by argument tuple."""

    def __init__(self, name, instances):
        self.__name__ = name
        self._instances = instances

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
        """A copy of {argument tuple: class or function} for every bound instantiation."""
        return dict(self._instances)

    def __instancecheck__(self, obj):
        return isinstance(obj, tuple(t for t in self._instances.values() if isinstance(t, type)))

    def __subclasscheck__(self, cls):
        return issubclass(cls, tuple(t for t in self._instances.values() if isinstance(t, type)))

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

deformation_gradient_sampling_response = _templates["deformation_gradient_sampling_response"]
value_sampling_response = _templates["value_sampling_response"]
gradient_sampling_response = _templates["gradient_sampling_response"]
hessian_sampling_response = _templates["hessian_sampling_response"]
DeformationGradientResponse = _native.DeformationGradientResponse
InterpolatedResponse = _templates["InterpolatedResponse"]

NeoHookean = _native.NeoHookean
KoiterStVK = _native.KoiterStVK

TriangleMidEdgeRestData = _native.TriangleMidEdgeRestData
TriangleMidEdgeKinematics = _native.TriangleMidEdgeKinematics
eval_triangle_mid_edge_kinematics = _native.eval_triangle_mid_edge_kinematics
eval_triangle_mid_edge_feasible_step = _native.eval_triangle_mid_edge_feasible_step
eval_positive_det_feasible_step = _native.eval_positive_det_feasible_step
eval_gradient_wrt_dofs = _native.eval_gradient_wrt_dofs
eval_hessian_wrt_dofs = _native.eval_hessian_wrt_dofs

VolumetricElement3D = _templates["VolumetricElement3D"]
BasisDofTransform = _templates["BasisDofTransform"]

VolumetricFemEnergy = _templates["VolumetricFemEnergy"]
VolumetricFemForceFpResponse = _templates["VolumetricFemForceFpResponse"]
VolumetricFemForceMaterialResponse = _templates["VolumetricFemForceMaterialResponse"]
ShellFemEnergy = _native.ShellFemEnergy
ShellFemForceMaterialResponse = _native.ShellFemForceMaterialResponse
ShellFemForceFpResponse = _native.ShellFemForceFpResponse

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
    "value_sampling_response",
    "deformation_gradient_sampling_response",
    "gradient_sampling_response",
    "hessian_sampling_response",
    "InterpolatedResponse",
    "DeformationGradientResponse",
    "NeoHookean",
    "KoiterStVK",
    "PyVolumetricMaterial",
    "PyShellMaterial",
    "TriangleMidEdgeRestData",
    "TriangleMidEdgeKinematics",
    "eval_triangle_mid_edge_kinematics",
    "eval_triangle_mid_edge_feasible_step",
    "eval_positive_det_feasible_step",
    "eval_gradient_wrt_dofs",
    "eval_hessian_wrt_dofs",
    "VolumetricElement3D",
    "BasisDofTransform",
    "VolumetricFemEnergy",
    "VolumetricFemForceMaterialResponse",
    "VolumetricFemForceFpResponse",
    "ShellFemEnergy",
    "ShellFemForceMaterialResponse",
    "ShellFemForceFpResponse",
]

LogDiagonalFpResponse = _native.LogDiagonalFpResponse
__all__ += ["LogDiagonalFpResponse"]

VolumetricBatchMaterial = _native.VolumetricBatchMaterial
ShellBatchMaterial = _native.ShellBatchMaterial
PointwiseVolumetricAdapter = _templates["PointwiseVolumetricAdapter"]
PointwiseShellAdapter = _templates["PointwiseShellAdapter"]
__all__ += ["VolumetricBatchMaterial", "ShellBatchMaterial", "PointwiseVolumetricAdapter", "PointwiseShellAdapter"]
