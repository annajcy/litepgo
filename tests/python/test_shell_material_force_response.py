import gc

import numpy as np
import pytest

from litepgo import dof, energy, fem, mesh, solver

Force = fem.ShellFemForceMaterialResponse
Energy = fem.ShellFemEnergy


def problem():
    X = np.array([[0., 0., 0.], [2., 0., 0.], [0., 1., 0.],
                  [2., 2., .1], [-1., 0., -.1], [0., -2., .2]])
    topology = mesh.TriangleTopology(6, np.array([[0, 1, 2], [2, 1, 3], [0, 2, 4], [1, 0, 5]], dtype=np.int32))
    stencil = mesh.make_triangle_mid_edge_stencils(topology)
    mapping = dof.StencilDofMap(6, stencil.vertices, 3)
    rest = [fem.TriangleMidEdgeRestData(mapping.gather(e, X.ravel()), stencil.opposite[e]) for e in range(4)]
    u = .03 * np.linspace(-1, 1, 18)
    u[[11, 14, 17]] += [.2, -.15, .18]
    return topology, mapping, rest, u, [.1, .2, .15, .3]


def observed(data, theta):
    _, mapping, rest, u, thickness = data
    parameters = [theta if theta.size == 2 else theta[2*e:2*e+2] for e in range(4)]
    return Energy(fem.PointwiseShellAdapter[fem.KoiterStVK](), mapping, rest, thickness, parameters).eval_gradient(u)


@pytest.mark.parametrize("batch_size", [1, 3, 256])
def test_shared_and_per_face_parameters_match_energy(batch_size):
    data = problem()
    topology, mapping, rest, u, thickness = data
    shared = Force(fem.PointwiseShellAdapter[fem.KoiterStVK](), dof.ConstantDofMap(topology, 2), mapping, rest, u, thickness, batch_size)
    element = Force(fem.PointwiseShellAdapter[fem.KoiterStVK](), dof.ElementDofMap(topology, 2), mapping, rest, u, thickness, batch_size)
    theta = np.array([2., 3.])
    per = np.array([2., 3., 4., 5., -.5, 2., 8., 9.])
    assert isinstance(shared, energy.MeshResponse)
    assert isinstance(shared, fem.ShellFemForceMaterialResponse)
    shared_evaluation = shared.evaluate(theta)
    J = shared_evaluation.jacobian()
    value = shared_evaluation.value
    element_evaluation = element.evaluate(per)
    E = element_evaluation.jacobian()
    per_value = element_evaluation.value
    np.testing.assert_allclose(value, observed(data, theta), atol=1e-13, rtol=1e-12)
    np.testing.assert_allclose(per_value, observed(data, per), atol=1e-13, rtol=1e-12)
    np.testing.assert_allclose(J.toarray(), sum(E[:, 2*e:2*e+2].toarray() for e in range(4)), atol=1e-13)
    for response, point in [(shared, theta), (element, per)]:
        derivative = response.evaluate(point).jacobian()
        d = np.sin(np.arange(response.num_inputs))
        w = np.cos(np.arange(response.num_outputs))
        np.testing.assert_allclose(response.evaluate(point).jvp(d), derivative @ d, rtol=1e-10, atol=1e-11)
        np.testing.assert_allclose(response.evaluate(point).vjp(w), derivative.T @ w, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(shared.evaluate(theta).value, value)
    with pytest.raises(ValueError):
        shared.evaluate(np.array([1.0, 0.0])).value
    np.testing.assert_array_equal(shared.evaluate(theta).jacobian().indices, J.indices)
    u[:] = 0
    thickness[:] = [100.] * 4
    rest.clear()
    del data, mapping, topology
    gc.collect()
    np.testing.assert_allclose(shared.evaluate(theta).value, value, atol=1e-13)




def test_material_mixed_derivative_and_parameter_recovery():
    a = np.array([[2.2, .4], [.4, 1.6]])
    b = np.array([[.3, -.1], [-.1, .15]])
    a_bar = np.array([[2., .3], [.3, 1.4]])
    b_bar = np.array([[.1, .03], [.03, -.2]])
    theta = np.array([2., 3.])
    J = fem.KoiterStVK.eval_d2W_dab_dtheta(a, b, a_bar, b_bar, .2, theta)
    assert J.shape == (6, 2)
    for j in range(2):
        plus, minus = theta.copy(), theta.copy()
        plus[j] += 1e-6
        minus[j] -= 1e-6
        fd = (fem.KoiterStVK.eval_dW_dab(a,b,a_bar,b_bar,.2,plus)
              - fem.KoiterStVK.eval_dW_dab(a,b,a_bar,b_bar,.2,minus))/2e-6
        np.testing.assert_allclose(J[:, j], fd, atol=1e-9)
    data = problem()
    topology, mapping, rest, u, thickness = data
    response = Force(fem.PointwiseShellAdapter[fem.KoiterStVK](), dof.ConstantDofMap(topology, 2), mapping, rest, u, thickness)
    target = observed(data, theta)

    class Residual(energy.FixedPatternResponse):
        def __init__(self):
            super().__init__(np.ones((response.num_outputs, response.num_inputs)))

        def eval_impl(self, x, need):
            evaluation = response.evaluate(x)
            jacobian = evaluation.jacobian() if need else None
            value = evaluation.value
            return (value-target, jacobian) if need else value-target

    result = solver.gauss_newton_solve(Residual(), np.array([.7, 1.2]),
                                     solver.GaussNewtonOptions(gradient_tolerance=1e-12))
    assert result.status == solver.GaussNewtonStatus.Converged
    np.testing.assert_allclose(result.x, theta, atol=1e-7, rtol=0)


def test_invalid_fixed_inputs_and_empty_mesh():
    topology, mapping, rest, u, thickness = problem()
    parameters = dof.ConstantDofMap(topology, 2)
    for args in [(parameters, mapping, [], u, thickness),
                 (parameters, mapping, rest, u[:-1], thickness),
                 (parameters, mapping, rest, u, []),
                 (parameters, mapping, rest, u, [0.]*4),
                 (dof.ConstantDofMap(topology, 3), mapping, rest, u, thickness)]:
        with pytest.raises(ValueError):
            Force(fem.PointwiseShellAdapter[fem.KoiterStVK](), *args)
    with pytest.raises(ValueError):
        Force(fem.PointwiseShellAdapter[fem.KoiterStVK](), parameters, mapping, rest, u, thickness, 0)
    empty = mesh.TriangleTopology(0, np.empty((0, 3), dtype=np.int32))
    zero = Force(fem.PointwiseShellAdapter[fem.KoiterStVK](), dof.ConstantDofMap(empty, 2), dof.StencilDofMap(0, [], 3), [], np.empty(0), [])
    zero_evaluation = zero.evaluate(np.array([2.0, 3.0]))
    J = zero_evaluation.jacobian()
    value = zero_evaluation.value
    assert value.size == 0 and J.shape == (0, 2)
