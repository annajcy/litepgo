# Sparse factorization lifecycle

A direct solver separates symbolic structure, numerical factorization, and
right-hand-side solves. Choose an appropriate [backend](backends.md) for the
matrix before starting this lifecycle.

## Factor and solve

For $Ax=b$, the lifecycle separates work that depends only on sparsity from work
that depends on numerical values.

```python
import numpy as np
import scipy.sparse as sp
from litepgo import linalg

A = sp.csc_matrix([[4., 1.], [1., 3.]])
b = np.array([1., 2.])
linear = linalg.AutoSparseSolver()
assert linear.analyze(A)
assert linear.factorize(A)
x = linear.solve(b)
np.testing.assert_allclose(A @ x, b, atol=1e-12)
```

`compute(A)` combines `analyze(A)` and `factorize(A)` for a one-off factorization.
For repeated matrices with the same pattern, analyze once, then factorize each
new set of values. Multiple right-hand sides can reuse one successful
factorization through repeated vector solves.

Native input matrices must be compressed CSC with sorted, unique row indices in
each column. Symmetric matrices store **both** triangles. Structural zeros still
belong to the pattern. The Python sparse argument is copied into native storage.
Symmetry and definiteness remain caller contracts.

`analyze` or `factorize` returns false on the corresponding backend failure;
invalid inputs or lifecycle misuse throw. A numerical factorization failure
retains the symbolic analysis, so a caller can change values, such as a diagonal
shift, and retry. A pattern change requires a new analysis. `solve` requires a
successful factorization. One solver object serves one caller at a time.
