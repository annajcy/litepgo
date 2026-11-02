# Hard constraints and reduced systems

Hard constraints prescribe scalar entries of the unknown vector. Reduction
selects the remaining free coordinates; the solver obtains fixed values from its
initial vector.

## Eliminate fixed DOFs

`DofReduction(n, fixed_indices)` stores a unique subset of scalar indices, not
prescribed values. Free indices are ordered increasingly. Let $P$ inject free
coordinates and $z_D$ contain prescribed entries. Let $q$ store free coordinates,
$g=\nabla E(z)$ be the full energy gradient, and $H=\nabla^2E(z)$ its Hessian. Then

$$
z=z_D+Pq,\qquad g_f=P^Tg,\qquad H_{ff}=P^THP.
$$

`newton_solve` takes prescribed values from its initial vector and keeps them
exactly. For a search direction, fixed entries are zero. Fixing all xyz entries
of a boundary vertex is different from fixing only its normal component.

The example uses four vertices with three displacement DOFs each:

```python
import numpy as np
from litepgo import dof

initial = np.zeros(12)
initial[:3] = [0.0, 0.0, 0.1]
fixed = dof.DofReduction(initial.size, [0, 1, 2])
# The solver will keep vertex 0 at this displacement, including its 0.1 z value.
```

C++ `reduce(full)` copies the free entries. `expand(reduced, full)` updates only
free entries, leaving the fixed entries of `full` as supplied. `ReducedMatrix`
refreshes a reduced sparse matrix without rebuilding its structural mapping.
