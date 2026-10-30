"""Locate independent points and interpolate physical DOFs for P1, Q1 and Hermite."""
import numpy as np
from litepgo import fem, dof, mesh


def run(basis, geometry, name, width):
    n = 4 if geometry is fem.TetP1Basis else 8
    X = np.array([geometry.reference_node(i) for i in range(n)]) * [2.,3.,4.]
    fields = [fem.InterpolatedField3D[geometry](X)]
    xi = np.array([[.1,.2,.3],[.2,.1,.2]])
    points = np.array([fields[0].eval_value(p) for p in xi])
    embedding = fem.ElementLocator[geometry](fields).locate(points)
    known = fem.PointEmbedding(np.zeros(2,dtype=np.int32),xi)
    topology = (mesh.TetTopology if n == 4 else mesh.HexTopology)(n,np.arange(n,dtype=np.int32).reshape(1,-1))
    layout = dof.VertexDofMap(topology,width)
    W = fem.build_point_displacement_map[basis](embedding,layout,fields)
    exact = fem.build_point_displacement_map[basis](known,layout,fields)
    u = np.linspace(-.2,.3,layout.num_dofs)
    coefficients = fem.BasisDofTransform[basis](fields[0]).to_coefficients(layout.gather(0,u))
    field = fem.InterpolatedField3D[basis](coefficients)
    kinematics = dof.SurfaceKinematics(points,W)
    expected = np.array([field.eval_value(p) for p in xi]).ravel()
    np.testing.assert_allclose(kinematics.displacements(u),expected,atol=1e-10)
    np.testing.assert_allclose(W@u,exact@u,atol=1e-10)
    print(f"{name}: points=2, verified=1")


def main():
    run(fem.TetP1Basis,fem.TetP1Basis,"TetP1",3)
    run(fem.HexQ1Basis,fem.HexQ1Basis,"HexQ1",3)
    run(fem.HexHermiteBasis,fem.HexQ1Basis,"HexHermite",24)


if __name__ == "__main__":
    main()
