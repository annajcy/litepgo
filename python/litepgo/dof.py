"""Degree-of-freedom maps and reductions, mirroring cpp/litepgo/dof."""

from litepgo._core import dof as _native

DofReduction = _native.DofReduction
VertexDofMap = _native.VertexDofMap
StencilDofMap = _native.StencilDofMap
ElementDofMap = _native.ElementDofMap
ConstantDofMap = _native.ConstantDofMap

__all__ = ["DofReduction", "VertexDofMap", "StencilDofMap", "ElementDofMap", "ConstantDofMap"]
