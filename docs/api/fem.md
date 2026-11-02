# FEM from Python

`litepgo.fem` mirrors the C++ API: the same classes, constructor arguments and
template arguments. A C++ class template is a `fem.Template`; indexing it with the
template arguments, in C++ order, returns the bound instantiation. Basis, quadrature
and material classes serve as type arguments and also expose their static
evaluations (`eval_N`, `points()`, `eval_W`, ...).

```python
import numpy as np
from litepgo import dof, energy, fem, mesh, solver

data = mesh.read_veg("assets/box-hex.veg")
n = data.topology.num_cells
rest_map = dof.VertexDofMap(data.topology, 3)
XField = fem.InterpolatedField3D[fem.HexQ1Basis]
X = [XField(rest_map.gather(e, data.positions.ravel())) for e in range(n)]
Fp = [fem.IdentityField3D(np.eye(3).ravel())] * n
theta = [fem.IdentityField3D(np.array([2.0, 3.0]))] * n
Energy = fem.VolumetricFemEnergy[fem.NeoHookean, fem.HexHermiteBasis, fem.GaussLegendre3D[4]]
elastic = Energy(dof.VertexDofMap(data.topology, 24), X, Fp, theta)
```

The building blocks are bound too: `LocalEnergy2D/3D`,
`VolumetricMaterialLocalEnergy{F,Fp,Theta}3D[M]` (which borrow their fields and keep
them alive), `ShellMaterialLocalEnergy2D[M]`, `VolumetricElement3D[Q]`,
`BasisDofTransform[B]` (physical DOFs to basis coefficients, e.g. Hermite slots), and the
kinematics and chain-rule functions. Where C++ fills optional output pointers or adds
into caller buffers (`eval`, `accumulate`, `eval_triangle_mid_edge_kinematics`), Python
takes `gradient=`/`hessian=` flags and returns `(value, gradient or None, hessian or None)`.
Matrix and vector arguments must be NumPy arrays.

The energies' field template arguments are erased (`fem::SharedField3D`): each
list may hold any `Field3D`, chosen per element, and the energy shares ownership
of them. Material, basis and quadrature arguments are compiled in;
`cpp/litepgo/bindings/fem/volumetric_fem_energy.cpp` and the other matching FEM
binding files list the available instantiations. Indexing any other combination
raises `TypeError` naming them. Add an instantiation in the corresponding file
to bind another. Fields and energies cannot be subclassed in Python: they are
evaluated per quadrature point inside parallel C++ loops. `examples/fem/**/*.py` mirror
the C++ FEM examples line by line, and `tests/python/test_fem.py` checks that
the forward pairs agree in connectivity and numerical output to the specified tolerances. `solver.newton_solve(energy, fixed, x0, options,
solver=None)` likewise follows the C++ signature (`fixed` is a `dof.DofReduction`;
the solver defaults to `AutoSparseSolver()`), and `solver.NewtonOptions()` starts from
the C++ defaults, with a NaN tolerance that must be set before solving.
