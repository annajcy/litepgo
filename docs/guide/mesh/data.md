# Mesh data and topology

Create a mesh from position and connectivity arrays. A mesh owns geometry; a
DOF map or material model is a separate object.

## Construct and inspect a mesh

```python
import numpy as np
from litepgo import mesh

positions = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
faces = np.array([[0, 1, 2]], dtype=np.int32)
topology = mesh.TriangleTopology(3, faces)
surface = mesh.TriangleMesh(positions, topology)
assert surface.positions.shape == (3, 3)
assert surface.cells.shape == (1, 3)
```

`TriangleMesh`, `TetMesh`, and `HexMesh` correspond to cells with 3, 4, and 8
vertices. `QuadTopology` also exists for parameter-domain data. Python positions
and cell properties return independent copies: construct a new mesh to change
its coordinates. In C++, `MeshData<K>` combines a packed xyz vector and
`MeshTopology<K>`; VEG input returns a tet/hex variant.
