# DOF maps

A DOF map gathers an ordered local vector from global scalar unknowns. Select
a map according to how unknowns are shared, rather than their physical units.

## Choose a map

| Map | Local layout | Use |
| --- | --- | --- |
| `VertexDofMap(topology, width)` | One row per cell vertex, `width` scalars per row | Nodal displacement, Hermite physical DOFs |
| `StencilDofMap(num_vertices, stencils, width)` | One row per listed stencil vertex | Shell curvature, edge energies, one-rings |
| `ElementDofMap(topology, width)` | Independent parameter block per element | Per-element material/plastic parameters |
| `ConstantDofMap(topology, width)` | The same parameter block for every element | Globally shared material/plastic parameters |

For vertex and stencil maps, scalar indices are `width*v + slot`. `gather(e, z)`
returns a local matrix; flatten it by row for differentiation. Stencils retain
caller order, must contain valid unique vertex IDs, and may have different sizes.

For example, gather one tetrahedron's xyz displacement:

```python
import numpy as np
from litepgo import dof, mesh

cells = np.array([[0, 1, 2, 3]], dtype=np.int32)
topology = mesh.TetTopology(4, cells)
mapping = dof.VertexDofMap(topology, 3)
u = np.arange(mapping.num_dofs, dtype=np.float64)
local = mapping.gather(0, u)
assert local.shape == (4, 3)
np.testing.assert_array_equal(local.ravel(), u)
```

`ConstantDofMap(topology, 2)` gives two global inputs regardless of element count.
`ElementDofMap(topology, 2)` gives two independent inputs per element. Input and
output maps of a `MeshResponse` must describe the same element ordering.
