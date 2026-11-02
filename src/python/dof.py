"""Degree-of-freedom maps and reductions, mirroring src/core/dof."""

from litepgo._core import dof as _native

DofReduction = _native.DofReduction
VertexDofMap = _native.VertexDofMap
StencilDofMap = _native.StencilDofMap
ElementDofMap = _native.ElementDofMap
ConstantDofMap = _native.ConstantDofMap
SurfaceKinematics = _native.SurfaceKinematics

build_vertex_displacement_map = _native.build_vertex_displacement_map

__all__ = ["build_vertex_displacement_map", "DofReduction", "VertexDofMap", "StencilDofMap", "ElementDofMap", "ConstantDofMap", "SurfaceKinematics"]
