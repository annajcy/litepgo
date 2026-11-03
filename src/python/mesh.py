"""Mesh connectivity, double-precision geometry, and file I/O, mirroring src/core/mesh.

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
make_triangle_mid_edge_stencils = _native.make_triangle_mid_edge_stencils
read_obj = _native.read_obj
write_obj = _native.write_obj
read_veg = _native.read_veg
write_veg = _native.write_veg
read_msh = _native.read_msh
write_msh = _native.write_msh
has_gmsh = _native.has_gmsh
has_cgal = _native.has_cgal
has_self_intersections = _native.has_self_intersections

ExtractedSurface = _native.ExtractedSurface
extract_surface_with_vertices = _native.extract_surface_with_vertices

__all__ = ["ExtractedSurface", "extract_surface_with_vertices",
    "TriangleTopology",
    "TetTopology",
    "QuadTopology",
    "HexTopology",
    "TriangleMesh",
    "TetMesh",
    "HexMesh",
    "TriangleMidEdgeStencils",
    "extract_surface",
    "make_triangle_mid_edge_stencils",
    "read_obj",
    "write_obj",
    "read_veg",
    "write_veg",
    "read_msh",
    "write_msh",
    "has_gmsh",
    "has_cgal",
    "has_self_intersections",
]
