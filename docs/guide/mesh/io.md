# I/O and surface extraction

Read geometry, preserve exact boundary vertex IDs, and export meshes. The
examples below assume a built litepgo package and the repository root.

## Read, write, and extract surfaces

```python
from pathlib import Path
from litepgo import mesh

volume = mesh.read_veg("assets/box-tet.veg")
output = Path("output/mesh-guide")
output.mkdir(parents=True, exist_ok=True)
mesh.write_veg(output / "copy.veg", volume)
mesh.write_obj(output / "boundary.obj", mesh.extract_surface(volume))
extracted = mesh.extract_surface_with_vertices(volume)
assert len(extracted.vertices) == extracted.mesh.topology.num_vertices
```

`extract_surface` triangulates the outward-wound boundary of a conforming tet/hex
mesh; hex boundary quads are split into triangles. It keeps boundary vertices in
original order. `extract_surface_with_vertices` also returns their exact volume
vertex IDs, needed for displacement maps without point-location roundoff.

| Format | Public functions | Scope |
| --- | --- | --- |
| OBJ | `read_obj`, `write_obj` | Triangle surface geometry |
| VEG | `read_veg`, `write_veg` | Text TET/CUBIC volume geometry; relative `*INCLUDE` with cycle detection |
| MSH | `read_msh`, `write_msh` | Optional Gmsh backend; homogeneous tet4/hex8; ASCII MSH 4.1 output |
| ABC | `write_abc_from_objs` | Optional Alembic Ogawa cache for a triangle mesh with fixed connectivity |

C++ I/O headers are `litepgo/mesh/io/obj.hpp`, `veg.hpp`, and optional `msh.hpp`.
Link `litepgo::mesh`, and `litepgo::mesh_gmsh` for MSH. Python exposes
`mesh.has_gmsh` to test the configured capability. Configure
`LITEPGO_WITH_GMSH=ON` to build it; the backend owns a serialized Gmsh session and
rejects an already initialized external session.

## Export an OBJ sequence

Build with `LITEPGO_WITH_ALEMBIC=ON` and check `mesh.has_alembic` in Python.
C++ callers include `litepgo/mesh/io/alembic.hpp` and link `litepgo::mesh_alembic`.

```python
from pathlib import Path
from litepgo import mesh

# These frame numbers are zero-padded, so name order matches playback order.
frames = sorted(Path("output/obj").glob("frame_*.obj"))
mesh.write_abc_from_objs("output/loading.abc", frames, fps=24)
```

The exporter reads the paths in the supplied order and writes one mesh sampled
from time zero at intervals of `1 / fps` seconds. Frames must have matching
vertex numbering and identical face indices in the same order. The first
frame supplies the topology and later frames update positions. The cache
stores float32 XYZ positions in the OBJ coordinate system, with faces wound
according to Alembic convention. A failed export preserves an existing destination.

## Geometry limitations and write behavior

Readers do not preserve UVs, normals, materials, physical groups, or boundary
labels. No welding, rescaling, orientation repair, or mesh generation is performed.
Supply triangulated OBJ for simulation; generic polygon triangulation is not
reliable for every nonconvex polygon. OBJ decimal parsing can introduce
floating-point rounding. Mixed/unsupported and empty MSH volume cells are rejected.

Writes replace the named file. Invalid mesh data is checked before opening it;
stream failures are reported. The examples create parent output directories
explicitly. Corner-only surface extraction does not visualize higher-order
Hermite deformation between vertices; use [boundary sampling](../fem/embedding.md).

For objectives acting directly on vertex positions, read the
[Laplacian and ARAP examples](../../examples/mesh/index.md). Those examples use generated
geometry and explicitly define their energies through `MeshEnergy`.
