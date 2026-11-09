from batch_fixtures import BatchResponseFixture
import numpy as np
import pytest
from scipy import sparse
from litepgo import dof, energy


class Rectangular(energy.FixedPatternResponse):
    def __init__(self):
        self.A = np.array([[1., 2.], [-3., 4.], [5., 0.]])
        super().__init__(sparse.csc_matrix(self.A))

    def eval_impl(self, x, need_jacobian):
        return (self.A @ x, self.A) if need_jacobian else self.A @ x


def check_products(r, x):
    J = r.eval_jacobian(x)
    dx = np.arange(r.num_inputs, dtype=float) + .3
    gy = np.arange(r.num_outputs, dtype=float) - .4
    a, b = r.eval_jvp(x, dx), r.eval_vjp(x, gy)
    np.testing.assert_allclose(a, J @ dx)
    np.testing.assert_allclose(b, J.T @ gy)
    np.testing.assert_allclose(gy @ a, dx @ b, atol=1e-12)
    eps = 1e-6
    np.testing.assert_allclose(a, (r.eval_value(x+eps*dx)-r.eval_value(x-eps*dx))/(2*eps), atol=1e-8)
    for call, n in [(r.eval_jvp, r.num_inputs), (r.eval_vjp, r.num_outputs)]:
        with pytest.raises(ValueError):
            call(x, np.zeros(n+1))
        with pytest.raises(ValueError):
            call(x, np.full(n, np.nan))


def test_base_fallback_and_override():
    check_products(Rectangular(), np.array([.2, .4]))
    class Direct(Rectangular):
        def eval_impl(self, x, need_jacobian):
            raise AssertionError('Jacobian path used')
        def eval_jvp_impl(self, x, dx):
            return self.A @ dx
        def eval_vjp_impl(self, x, gy):
            return self.A.T @ gy
    r = Direct()
    np.testing.assert_allclose(r.eval_jvp(np.zeros(2), np.ones(2)), [3, 1, 5])
    np.testing.assert_allclose(r.eval_vjp(np.zeros(2), np.ones(3)), [3, 6])


@pytest.mark.parametrize('batch_size', [1, 2, 256])
def test_mesh_fallback_and_direct_products(batch_size):
    inp = dof.StencilDofMap(3, [[0, 1], [0, 2], [1, 2]], 1)
    out = dof.StencilDofMap(4, [[0, 1, 2], [1, 2, 3], [0, 2, 3]], 1)
    A = np.array([[1., 2.], [3., -4.], [-2., 5.]])
    def batch(ids, x, need_jacobian):
        values = (x*x) @ A.T
        J = (A[None] * (2*x[:,None,:])).reshape(-1, 2)
        return (values, J) if need_jacobian else values
    r = BatchResponseFixture(inp, out, batch, batch_size)
    x = np.array([.2, -.5, .7])
    check_products(r, x)
    class Direct(energy.MeshResponse):
        def __init__(self):
            super().__init__(inp, out, batch_size)
        def eval_batch(self, ids, x, need_jacobian):
            raise AssertionError('Jacobian path used')
        def eval_jvp_batch(self, ids, x, dx):
            return (2*x*dx) @ A.T
        def eval_vjp_batch(self, ids, x, gy):
            return 2*x*(gy @ A)
    direct = Direct()
    np.testing.assert_allclose(direct.eval_jvp(x, x), r.eval_jvp(x,x))
    np.testing.assert_allclose(direct.eval_vjp(x, np.ones(4)), r.eval_vjp(x,np.ones(4)))
    class Fallback(Direct):
        def eval_batch(self, ids, x, need_jacobian):
            return batch(ids, x, need_jacobian)
    check_products(Fallback(), x)
    class Default(energy.MeshResponse):
        def __init__(self):
            super().__init__(inp,out,batch_size)
        def eval_batch(self, ids, x, need_jacobian):
            return batch(ids,x,need_jacobian)
    check_products(Default(),x)


def test_mesh_energy_subclass():
    mapping = dof.StencilDofMap(2, [[0],[1]], 1)
    class Quadratic(energy.MeshEnergy):
        def __init__(self):
            super().__init__(mapping,1)
        def eval_batch(self, ids, x, g, h):
            return .5*(x*x).sum(1), x if g else None, np.ones_like(x) if h else None
        def eval_feasible_step_batch(self, ids, x, dx):
            return np.full(len(ids), .25)
    obj = Quadratic()
    x = np.array([2.,3.])
    assert obj.eval_value(x) == 6.5
    np.testing.assert_array_equal(obj.eval_gradient(x),x)
    np.testing.assert_array_equal(obj.eval_hessian(x).toarray(),np.eye(2))
    assert obj.max_feasible_step(x,x) == .25


def test_empty_mesh_products():
    mapping = dof.StencilDofMap(2, [], 1)
    def forbidden(*args):
        raise AssertionError('empty mesh called evaluator')
    r = BatchResponseFixture(mapping, mapping, forbidden)
    np.testing.assert_array_equal(r.eval_jvp(np.ones(2),np.ones(2)),np.zeros(2))
    np.testing.assert_array_equal(r.eval_vjp(np.ones(2),np.ones(2)),np.zeros(2))


@pytest.mark.parametrize('ni, no', [(0, 0), (0, 2), (2, 0)])
def test_empty_local_product_blocks(ni, no):
    inp = dof.StencilDofMap(3, [list(range(ni))] * 3, 1)
    out = dof.StencilDofMap(4, [list(range(no))] * 3, 1)

    def batch(ids, x, need):
        values = np.zeros((len(ids), no))
        return (values, np.zeros((len(ids) * no, ni))) if need else values

    check_products(BatchResponseFixture(inp, out, batch, 2), np.ones(3))


def test_invalid_product_and_recovery():
    mapping = dof.StencilDofMap(1, [[0]], 1)
    class Bad(energy.MeshResponse):
        def __init__(self):
            super().__init__(mapping,mapping)
            self.mode = 'nan'
        def eval_jvp_batch(self, ids, x, dx):
            if self.mode == 'nan':
                return np.full_like(dx,np.nan)
            if self.mode == 'shape':
                return np.zeros((len(ids),2))
            if self.mode == 'throw':
                raise KeyError('failure')
            return dx
    r=Bad()
    for mode, error in [('nan',RuntimeError),('shape',RuntimeError),('throw',KeyError)]:
        r.mode=mode
        with pytest.raises(error):
            r.eval_jvp(np.ones(1),np.ones(1))
    r.mode='ok'
    np.testing.assert_array_equal(r.eval_jvp(np.ones(1),np.ones(1)),np.ones(1))


def test_dynamic_patterns_and_mixed_mesh_shapes():
    from test_dynamic_response import Piecewise
    r = Piecewise()
    for x in [np.array([.3,.4]), np.array([-.3,.4])]:
        check_products(r,x)
    inp = dof.StencilDofMap(3, [[0], [0,1], [2]], 1)
    out = dof.StencilDofMap(4, [[0,1], [1,2,3], [3,0]], 1)
    widths = [2,3,2]
    def batch(ids,x,need):
        no=widths[ids[0]]
        values=np.repeat((x*x).sum(1)[:,None],no,axis=1)
        J=np.repeat((2*x)[:,None,:],no,axis=1).reshape(-1,x.shape[1])
        return (values,J) if need else values
    check_products(BatchResponseFixture(inp,out,batch,2), np.array([.2,.3,-.4]))


def test_response_super_products():
    from test_dynamic_response import Piecewise

    class Conditional(Rectangular):
        direct = False

        def eval_jvp_impl(self, x, dx):
            if self.direct:
                return self.A @ dx
            return super().eval_jvp_impl(x, dx)

        def eval_vjp_impl(self, x, gy):
            if self.direct:
                return self.A.T @ gy
            return super().eval_vjp_impl(x, gy)

    r = Conditional()
    for direct in [False, True, False]:
        r.direct = direct
        check_products(r, np.array([.2, .4]))
    with pytest.raises(ValueError):
        r.eval_jvp_impl(np.zeros(2), np.ones(3))

    class Dynamic(Piecewise):
        def eval_jvp_impl(self, x, dx):
            return super().eval_jvp_impl(x, dx)

        def eval_vjp_impl(self, x, gy):
            return super().eval_vjp_impl(x, gy)

    r = Dynamic()
    for x in [np.array([.3, .4]), np.array([-.3, .4])]:
        check_products(r, x)


def test_mesh_super_products_and_invalid_batches():
    inp = dof.StencilDofMap(3, [[0, 1], [1, 2]], 1)
    out = dof.StencilDofMap(4, [[0, 1, 2], [1, 2, 3]], 1)
    A = np.array([[1., 2.], [3., -4.], [-2., 5.]])

    class Conditional(energy.MeshResponse):
        direct = False

        def __init__(self):
            super().__init__(inp, out, 1)

        def eval_batch(self, ids, x, need):
            values = x @ A.T
            return (values, np.tile(A, (len(ids), 1))) if need else values

        def eval_jvp_batch(self, ids, x, dx):
            if self.direct:
                return dx @ A.T
            return super().eval_jvp_batch(ids, x, dx)

        def eval_vjp_batch(self, ids, x, gy):
            if self.direct:
                return gy @ A
            return super().eval_vjp_batch(ids, x, gy)

    r = Conditional()
    for direct in [False, True, False]:
        r.direct = direct
        check_products(r, np.array([.2, -.5, .7]))
    for ids, x, d in [
        ([], np.empty((0, 2)), np.empty((0, 2))),
        ([-1], np.ones((1, 2)), np.ones((1, 2))),
        ([2], np.ones((1, 2)), np.ones((1, 2))),
        ([0], np.ones((1, 2)), np.ones((1, 3))),
        ([0], np.full((1, 2), np.nan), np.ones((1, 2))),
    ]:
        with pytest.raises(ValueError):
            r.eval_jvp_batch(ids, x, d)
    check_products(r, np.array([.2, -.5, .7]))


def test_mesh_energy_super_step_and_fem_inheritance():
    from litepgo import fem

    class Quadratic(energy.MeshEnergy):
        def __init__(self):
            super().__init__(dof.StencilDofMap(2, [[0], [1]], 1))

        def eval_feasible_step_batch(self, ids, x, dx):
            return .5 * super().eval_feasible_step_batch(ids, x, dx)

    obj = Quadratic()
    assert obj.max_feasible_step(np.ones(2), np.ones(2)) == .5
    assert issubclass(
        fem.VolumetricFemEnergy[fem.TetP1Basis, fem.TetCentroidQuadrature],
        energy.MeshEnergy,
    )
    assert issubclass(fem.ShellFemEnergy, energy.MeshEnergy)


def test_value_reentry_without_shared_workspace():
    class Reentrant(Rectangular):
        def eval_impl(self, x, need):
            if need:
                return self.eval_value(x), self.A
            return super().eval_impl(x, False)

    check_products(Reentrant(), np.array([.2, .4]))


def test_direct_products_accept_infeasible_none():
    import numpy as np
    import pytest
    from scipy.sparse import eye
    from litepgo import energy

    class Infeasible(energy.FixedPatternResponse):
        def __init__(self):
            super().__init__(eye(1, format="csc"))

        def eval_impl(self, x, need_jacobian):
            return None

        def eval_jvp_impl(self, x, dx):
            return None

        def eval_vjp_impl(self, x, gy):
            return None

    model = Infeasible()
    for operation in (model.eval_jvp, model.eval_vjp):
        with pytest.raises(ValueError, match="infeasible"):
            operation(np.ones(1), np.ones(1))


def test_mesh_products_accept_infeasible_none():
    from litepgo import dof, mesh

    class Infeasible(energy.MeshResponse):
        def eval_batch(self, ids, x, need_jacobian):
            return None

        def eval_jvp_batch(self, ids, x, dx):
            return None

        def eval_vjp_batch(self, ids, x, gy):
            return None

    topology = mesh.TetTopology(4, np.array([[0, 1, 2, 3]], dtype=np.int32))
    mapping = dof.ConstantDofMap(topology, 1)
    model = Infeasible(mapping, mapping)
    for operation in (model.eval_jvp, model.eval_vjp):
        with pytest.raises(ValueError, match="infeasible"):
            operation(np.ones(1), np.ones(1))
