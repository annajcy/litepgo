"""Double-precision mesh geometry I/O. Array properties return independent copies."""

from litepgo._core.io import TriangleMesh, TetMesh, HexMesh, read_obj, write_obj, read_veg, write_veg, read_msh, write_msh, has_gmsh

__all__ = ["TriangleMesh", "TetMesh", "HexMesh", "read_obj", "write_obj", "read_veg", "write_veg", "read_msh", "write_msh", "has_gmsh"]
