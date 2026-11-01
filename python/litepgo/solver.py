"""Nonlinear solvers, mirroring cpp/litepgo/solver.

Gauss-Newton accepts a Response and uses sparse damped normal equations for damped steps.
Jacobian patterns are updated at current iterates; symbolic analysis is reused
while the structure is unchanged. Trials evaluate only the residual vector.
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
