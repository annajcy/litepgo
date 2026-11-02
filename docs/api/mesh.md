# Mesh I/O

The `litepgo::io` CMake target provides `litepgo::mesh::read_obj/write_obj`
and `read_veg/write_veg`. Include `litepgo/mesh/io/obj.hpp` or `veg.hpp`.
The mesh module's `litepgo/mesh/extract_surface.hpp` provides `litepgo::mesh::extract_surface(mesh)`:
the boundary of a conforming tet or hex mesh as outward-wound triangles (hex quads are split), keeping only
boundary vertices in their original order. Write it with `write_obj` to view a
volume mesh as a surface.
`MeshData<K>` owns vertex-major xyz doubles and a `MeshTopology<K>`;
`TriangleMesh`, `TetMesh`, `HexMesh` name the supported shapes.
VEG readers return `VolumeMesh`, a tet/hex variant.

Enable `-DLITEPGO_WITH_GMSH=ON` for the separate `litepgo::io_gmsh`
target and `litepgo/mesh/io/msh.hpp`. CMake FetchContent downloads and builds pinned Gmsh 4.13.1 sources.
Tinyobjloader v2.0.0rc13 is also fetched and built with double precision.
The default build does not require Gmsh. MSH uses Gmsh for input and writes
ASCII MSH 4.1. Only homogeneous tet4 or hex8 volume meshes are accepted;
unsupported/mixed volume cells and empty volume meshes are rejected.
The backend owns a serialized Gmsh session and rejects an already initialized
external session rather than destroying its state.

Python exposes the same operations, with the topologies, through `from litepgo import mesh`:

```python
data = mesh.read_veg("model.veg")
assert data.positions.dtype.name == "float64"
mesh.write_veg("copy.veg", data)
mesh.write_obj("surface.obj", mesh.extract_surface(data))
data = mesh.TetMesh(data.positions, data.topology)
# mesh.has_gmsh reports whether read_msh/write_msh are enabled.
```

Python positions have shape `(n, 3)` and cells `(m, 3/4/8)`; properties
return independent copies. Construct a new mesh to change geometry.
Readers/writers preserve geometry, not materials, UVs, normals, physical groups,
or boundary labels. OBJ polygons are triangulated by tinyobjloader; its default
triangulation is not guaranteed for arbitrary complex polygons. Supply triangulated
OBJ for simulation. OBJ decimal parsing can introduce double-level rounding.
VEG supports text TET/CUBIC and relative `*INCLUDE`, with cycle detection.
No welding, rescaling, orientation repair, or mesh generation is performed.
FEM performs its own geometric validity checks. Writes replace the named file;
invalid mesh data is rejected before opening it, and stream errors are reported.

Run `tests/python/test_mesh.py` for precision, real-asset roundtrip, sparse MSH tags,
and invalid-input checks.
