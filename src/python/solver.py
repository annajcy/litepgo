"""Nonlinear solvers, mirroring src/core/solver.

Gauss-Newton accepts a Response and uses sparse damped normal equations for damped steps.
Jacobian patterns are updated at current iterates; symbolic analysis is reused
while the structure is unchanged. Trials evaluate only the residual vector.

StaticEquilibriumResponse(model, fixed, initial, options) takes a ParametricEnergy.
Its energy_at(p) view supplies Newton; force_at(u) supplies parameter derivatives.
Fixed DOFs and values must be parameter-independent. Each call starts from initial;
sensitivities use the exact unshifted Hessian. JVP/VJP delegate parameter products
to the model and need not construct either numerical Jacobian. Patterns remain explicit.
Nonconverged equilibria raise ValueError. See examples/fem/fitting/volume_equilibrium_material_fit.py.

"""

from litepgo._core import solver as _native

NewtonOptions = _native.NewtonOptions
NewtonStatus = _native.NewtonStatus
NewtonStep = _native.NewtonStep
NewtonResult = _native.NewtonResult
newton_solve = _native.newton_solve

__all__ = ["NewtonOptions", "NewtonStatus", "NewtonStep", "NewtonResult", "newton_solve"]

GaussNewtonOptions = _native.GaussNewtonOptions
GaussNewtonStatus = _native.GaussNewtonStatus
GaussNewtonResult = _native.GaussNewtonResult
gauss_newton_solve = _native.gauss_newton_solve

__all__ += ["GaussNewtonOptions", "GaussNewtonStatus", "GaussNewtonResult", "gauss_newton_solve"]

StaticEquilibriumResponse = _native.StaticEquilibriumResponse

__all__ += ["StaticEquilibriumResponse"]
