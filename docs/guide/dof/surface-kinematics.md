# Surface kinematics

Map a simulation displacement vector to the positions required by a surface
consumer such as contact. Let $u\in\mathbb R^n$ be displacement, $X_s\in\mathbb R^{3m}$
packed rest surface positions, and $W\in\mathbb R^{3m\times n}$ the sparse map.

## Map a surface to simulation unknowns

`SurfaceKinematics(rest_positions, W)` represents

$$
p(u)=X_s+Wu,\qquad \delta p=W\delta u.
$$

For example, map simulation vertices 0 and 2 to a two-vertex surface. The
constructor takes rest positions as an `(m, 3)` array; the formula above uses
their flattened representation.

```python
import numpy as np
from litepgo import dof

rest = np.array([[0., 0., 0.], [1., 0., 0.]])
W = dof.build_vertex_displacement_map([0, 2], num_dofs=9)
surface = dof.SurfaceKinematics(rest, W)
u = np.zeros(9)
u[3 * 2 + 1] = 0.25
np.testing.assert_allclose(surface.positions(u).reshape(-1, 3),
                           [[0., 0., 0.], [1., 0.25, 0.]])
```

`positions(u)` returns a packed xyz vector with `3*m` entries.

For an extracted volume boundary, preserve exact vertex IDs with
`extract_surface_with_vertices`, then build `W` with
`build_vertex_displacement_map(vertices, num_dofs, dofs_per_vertex=3)`.
Hermite uses width 24; that exact-vertex map selects only the displacement-value
slots. For independent points and between-corner Hermite samples, build `W`
from a [point embedding](../fem/embedding.md).

`SurfaceKinematics` supplies positions, displacements, and the sparse map; it does
not itself define contact or interpolate a new geometry model. Combine it with
`AffineMappedEnergy` as explained in [contact](../contact/index.md).
