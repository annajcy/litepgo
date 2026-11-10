"""Precision diagnostics must not replace convergence or feasibility outcomes."""
from litepgo.solver import AbsoluteConvergence
import numpy as np
import pytest
import scipy.sparse as sp
from litepgo import dof, energy, solver

EPS = np.finfo(float).eps

class Scalar(energy.FixedPatternEnergy):
    def __init__(self, mode='floor', bound=1.):
        super().__init__(sp.eye(1, format='csc'))
        self.mode, self.bound = mode, bound

    def eval_values(self, x, need_gradient, need_hessian):
        if self.mode == 'quartic':
            v, g, h = x[0]**4/4, x[0]**3, 3*x[0]**2
        elif self.mode == 'oscillation':
            # Emulate a bounded deterministic gradient rounding error atop a quadratic.
            v, g, h = 1e6 + .5e12*x[0]**2, 1e12*(x[0] + (.25*EPS if x[0]>=0 else -.25*EPS)), 1e12
        else:
            h = -1e12 if self.mode == 'indefinite' else 1e12
            r = (x[0]-1.)-(0.4*EPS if self.mode in ('floor', 'indefinite') else .25)
            v, g = .5*h*r*r, h*r
        return v, np.array([g]) if need_gradient else None, np.array([[h]]) if need_hessian else None

    def feasible_step(self, x, dx):
        return 1. if not np.any(dx) else self.bound


def run(model, x=1., tolerance=1e-5, **kw):
    opts = dict(max_iterations=100, project_hessian=False,
                detect_precision_stagnation=True, precision_coordinate_scale=1.)
    opts.update(kw)
    steps=[]
    result=solver.newton_solve(model,dof.DofReduction(1,[]),np.array([x]),solver.NewtonOptions(**opts),on_iteration=steps.append, convergence=solver.AbsoluteConvergence(tolerance))
    return result,steps


def test_unrepresentable_full_step_is_precision_not_convergence():
    r,steps=run(Scalar())
    assert r.status==solver.NewtonStatus.PrecisionStagnation
    assert r.gradient_norm>1e-5 and r.linear_backward_error<1e-12
    assert r.scaled_direction_norm<=128*EPS and r.precision_stagnation_steps==1
    assert r.x[0]==1. and r.iterations==0 and steps[-1].step==0.


def test_opt_in_preserves_old_outcome():
    r,_=run(Scalar(),detect_precision_stagnation=False)
    assert r.status==solver.NewtonStatus.StepTooSmall


def test_roundoff_oscillation_uses_multiple_full_steps():
    r,_=run(Scalar('oscillation'),x=0.)
    assert r.status==solver.NewtonStatus.PrecisionStagnation
    assert r.precision_stagnation_steps==3 and r.iterations>=3


@pytest.mark.parametrize('bound',[0.,1e-20])
@pytest.mark.parametrize('mode',['floor','regular'])
def test_ccd_restriction_is_not_precision(bound,mode):
    r,_=run(Scalar(mode,bound))
    assert r.status==solver.NewtonStatus.StepTooSmall


def test_coordinate_cap_is_not_precision():
    r,_=run(Scalar('regular'),max_coordinate_step=1e-20)
    assert r.status==solver.NewtonStatus.StepTooSmall


def test_shifted_direction_is_not_precision():
    r,steps=run(Scalar('indefinite'))
    assert any(step.diagonal_shifts>0 for step in steps)
    assert r.status!=solver.NewtonStatus.PrecisionStagnation


def test_projected_direction_is_not_precision():
    r,_=run(Scalar(),project_hessian=True)
    assert r.status==solver.NewtonStatus.StepTooSmall


def test_normal_convergence_has_priority():
    r,_=run(Scalar('regular'))
    assert r.status==solver.NewtonStatus.Converged and r.x[0]==1.25
    r,_=run(Scalar(),tolerance=1e-3)
    assert r.status==solver.NewtonStatus.Converged and r.iterations==0


def test_tiny_but_improving_steps_continue_to_convergence():
    r,_=run(Scalar('quartic'),x=1e-15,tolerance=1e-60)
    assert r.status==solver.NewtonStatus.Converged and r.iterations>3


@pytest.mark.parametrize('scale',[0.,-1.,np.inf,np.nan])
def test_invalid_scale_is_rejected(scale):
    with pytest.raises(ValueError,match='precision_coordinate_scale'):
        run(Scalar(),precision_coordinate_scale=scale)


@pytest.mark.parametrize('budget', [1, 100])
def test_real_ipc_ccd_restriction_is_not_precision(budget):
    from litepgo import contact, mesh
    positions = np.array([[0., 0., 0.], [10., 0., 0.], [0., 10., 0.],
                          [2., 2., .5], [6., 2., 4.5], [2., 6., 4.5]])
    surface = mesh.TriangleMesh(positions, np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32))
    barrier = contact.IPCContactEnergy(surface, contact.IPCContactOptions(dhat=.1), contact.SpatialHash())
    x = positions.ravel(); target = x.copy(); target[11] = -.5
    potential = energy.WeightedSumEnergy([(energy.QuadraticEnergy(sp.eye(18, format='csc'), -target), 1.),
                                          (barrier, .001)])
    fixed = dof.DofReduction(18, [i for i in range(18) if i != 11])
    steps = []
    result = solver.newton_solve(potential, fixed, x,
        solver.NewtonOptions(max_iterations=budget,
                             project_hessian=False, detect_precision_stagnation=True),
        on_iteration=steps.append, convergence=AbsoluteConvergence(1e-8))
    assert 0 < steps[0].feasible_step < 1
    assert result.status == (solver.NewtonStatus.MaxIterations if budget == 1 else solver.NewtonStatus.Converged)
    assert result.x[11] > 0
