from litepgo.energy import AffineResponse
import gc

import numpy as np
import pytest

from litepgo import dof, energy, fem, mesh
from test_shell_material_force_response import problem

Force = fem.ShellFemForceFpResponse[fem.KoiterStVK]
P = np.array([[1.12, .07, -.02], [.03, .94, .06], [.01, -.03, 1.08]])


def explicit_force(data, plastic, theta):
    _, mapping, rest, u, thickness = data
    result = np.zeros(mapping.num_dofs)
    for e, reference in enumerate(rest):
        p = plastic if plastic.size == 9 else plastic[9*e:9*e+9]
        Y = reference.X_local @ p.reshape(3, 3, order="F").T
        natural = fem.TriangleMidEdgeRestData(Y, reference.opposite)
        # Keep x = X+u, and integrate with ORIGINAL area, not the transformed area.
        current, J, _ = fem.eval_triangle_mid_edge_kinematics(
            reference.X_local + mapping.gather(e, u), reference.opposite, gradient=True
        )
        inputs = (current.a, current.b, natural.a_bar, natural.b_bar, thickness[e], theta[e])
        result[np.array(mapping.dof_indices(e))] += (
            .5 * reference.sqrt_det_a_bar * J.T @ fem.KoiterStVK.eval_dW_dab(*inputs)
        )
    return result


@pytest.mark.parametrize("batch_size", [1, 3, 256])
def test_shared_and_element_fp_jacobians_and_original_area(batch_size):
    data = problem()
    topology, mapping, rest, u, thickness = data
    theta = [np.array([2.+e, 3.]) for e in range(4)]
    shared = Force(dof.ConstantDofMap(topology, 9), mapping, rest, u, thickness, theta, batch_size)
    element = Force(dof.ElementDofMap(topology, 9), mapping, rest, u, thickness, theta, batch_size)
    packed = P.ravel(order="F")
    per = np.concatenate([(1+.02*e)*packed for e in range(4)])
    assert isinstance(shared, energy.MeshResponse) and isinstance(shared, fem.ShellFemForceFpResponse)
    value, J = shared.eval_value_and_jacobian(packed)
    per_value, E = element.eval_value_and_jacobian(per)
    np.testing.assert_allclose(value, explicit_force(data, packed, theta), rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(per_value, explicit_force(data, per, theta), rtol=1e-11, atol=1e-12)
    for response, point, derivative in [(shared, packed, J), (element, per, E)]:
        for j in range(point.size):
            plus, minus = point.copy(), point.copy()
            plus[j] += 1e-6
            minus[j] -= 1e-6
            fd = (response.eval_value(plus)-response.eval_value(minus))/2e-6
            np.testing.assert_allclose(derivative.toarray()[:, j], fd, atol=2e-7, rtol=1e-6)
    repeated_J = element.eval_jacobian(np.tile(packed, 4)).toarray()
    np.testing.assert_allclose(J.toarray(), sum(repeated_J[:, 9*e:9*e+9] for e in range(4)), atol=1e-12)
    identity = np.eye(3).ravel(order="F")
    original = fem.ShellFemEnergy[fem.KoiterStVK](mapping, rest, thickness, theta)
    np.testing.assert_allclose(shared.eval_value(identity), original.eval_gradient(u), atol=1e-12)
    with pytest.raises(ValueError):
        shared.eval_value(np.zeros(9))
    np.testing.assert_array_equal(shared.eval_jacobian(packed).indices, J.indices)
    u[:] = 0
    rest.clear()
    theta[0][:] = 100
    del data, mapping, topology
    gc.collect()
    np.testing.assert_allclose(shared.eval_value(packed), value, atol=1e-12)




def test_natural_form_mixed_derivative_binding():
    a = np.array([[2.2, .4], [.4, 1.6]])
    b = np.array([[.3, -.1], [-.1, .15]])
    A = np.array([[2., .3], [.3, 1.4]])
    B = np.array([[.1, .03], [.03, -.2]])
    theta = np.array([2., 3.])
    mixed = fem.KoiterStVK.eval_d2W_dab_dabarbbar(a, b, A, B, .2, theta)
    assert mixed.shape == (6, 6)
    for j in range(6):
        plus, minus = [A.copy(), B.copy()], [A.copy(), B.copy()]
        r, c = [(0, 0), (0, 1), (1, 1)][j % 3]
        plus[j//3][r, c] += 1e-6
        minus[j//3][r, c] -= 1e-6
        if r != c:
            plus[j//3][c, r] += 1e-6
            minus[j//3][c, r] -= 1e-6
        fd = (fem.KoiterStVK.eval_dW_dab(a, b, *plus, .2, theta)
              - fem.KoiterStVK.eval_dW_dab(a, b, *minus, .2, theta))/2e-6
        np.testing.assert_allclose(mixed[:, j], fd, atol=1e-9)


def test_invalid_inputs_and_empty_mesh():
    topology, mapping, rest, u, thickness = problem()
    theta = [np.array([2., 3.]) for _ in rest]
    params = dof.ConstantDofMap(topology, 9)
    for args in [(params, mapping, [], u, thickness, theta),
                 (params, mapping, rest, u, [], theta),
                 (params, mapping, rest, u, thickness, []),
                 (params, mapping, rest, u[:-1], thickness, theta),
                 (dof.ConstantDofMap(topology, 8), mapping, rest, u, thickness, theta)]:
        with pytest.raises(ValueError):
            Force(*args)
    response = Force(params, mapping, rest, u, thickness, theta)
    for packed in [np.zeros(9), np.full(9, np.nan), np.diag([-1., 1., 1.]).ravel(), np.ones(8)]:
        with pytest.raises(ValueError):
            response.eval_value(packed)
    empty = mesh.TriangleTopology(0, np.empty((0, 3), dtype=np.int32))
    zero = Force(dof.ConstantDofMap(empty, 9), dof.StencilDofMap(0, [], 3), [], np.empty(0), [], [])
    value, J = zero.eval_value_and_jacobian(np.eye(3).ravel(order="F"))
    assert value.size == 0 and J.shape == (0, 9)


def test_shell_energy_fixed_fp_derivatives_and_validation():
    topology, mapping, rest, u, thickness = problem()
    theta = [np.array([2., 3.])] * len(rest)
    Energy = fem.ShellFemEnergy[fem.KoiterStVK]
    plastic = [P.copy() for _ in rest]
    elastic = Energy(mapping, rest, thickness, theta, Fp=plastic)
    original = Energy(mapping, rest, thickness, theta)
    identity = Energy(mapping, rest, thickness, theta, [np.eye(3)]*len(rest))
    np.testing.assert_allclose(identity.eval_gradient(u), original.eval_gradient(u), atol=1e-12)
    response = Force(dof.ConstantDofMap(topology, 9), mapping, rest, u, thickness, theta)
    np.testing.assert_allclose(elastic.eval_gradient(u), response.eval_value(P.ravel(order="F")), atol=1e-12)
    expected = explicit_force((topology, mapping, rest, u, thickness), P.ravel(order="F"), theta)
    np.testing.assert_allclose(elastic.eval_gradient(u), expected, atol=1e-12)
    H = elastic.eval_hessian(u).toarray()
    g = elastic.eval_gradient(u)
    for j in range(len(u)):
        plus, minus = u.copy(), u.copy()
        plus[j] += 1e-6
        minus[j] -= 1e-6
        np.testing.assert_allclose(g[j], (elastic.eval_value(plus)-elastic.eval_value(minus))/2e-6, atol=1e-8)
        np.testing.assert_allclose(H[:, j], (elastic.eval_gradient(plus)-elastic.eval_gradient(minus))/2e-6,
                                   atol=1e-7, rtol=1e-6)
    with pytest.raises(ValueError):
        Energy(mapping, rest, thickness, theta, plastic[:1])
    for value in (0., -1., np.nan, np.inf):
        invalid = [np.eye(3) for _ in rest]
        invalid[0][0, 0] = value
        with pytest.raises(ValueError):
            Energy(mapping, rest, thickness, theta, invalid)
    plastic[0][:] = 0
    rest.clear()
    np.testing.assert_array_equal(elastic.eval_gradient(u), g)
