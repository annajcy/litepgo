"""Mesh connectivity, double-precision geometry, and file I/O, mirroring cpp/litepgo/mesh.

Array properties return independent copies; construct a new mesh to change geometry.
"""

from litepgo._core import mesh as _native

TriangleTopology = _native.TriangleTopology
TetTopology = _native.TetTopology
QuadTopology = _native.QuadTopology
HexTopology = _native.HexTopology
TriangleMesh = _native.TriangleMesh
TetMesh = _native.TetMesh
HexMesh = _native.HexMesh
TriangleMidEdgeStencils = _native.TriangleMidEdgeStencils
extract_surface = _native.extract_surface
ExtractedSurface = _native.ExtractedSurface
extract_surface_with_vertices = _native.extract_surface_with_vertices
make_triangle_mid_edge_stencils = _native.make_triangle_mid_edge_stencils
read_obj = _native.read_obj
write_obj = _native.write_obj
read_veg = _native.read_veg
write_veg = _native.write_veg
read_msh = _native.read_msh
write_msh = _native.write_msh
has_gmsh = _native.has_gmsh

__all__ = [
    "TriangleTopology",
    "TetTopology",
    "QuadTopology",
    "HexTopology",
    "TriangleMesh",
    "TetMesh",
    "HexMesh",
    "TriangleMidEdgeStencils",
    "extract_surface",
    "ExtractedSurface",
    "extract_surface_with_vertices",
    "make_triangle_mid_edge_stencils",
    "read_obj",
    "write_obj",
    "read_veg",
    "write_veg",
    "read_msh",
    "write_msh",
    "has_gmsh",
]
