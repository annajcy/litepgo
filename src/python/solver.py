"""Nonlinear solvers, mirroring src/core/solver.

Gauss-Newton accepts a Response and uses sparse damped normal equations for damped steps.
Jacobian patterns are updated at current iterates; symbolic analysis is reused
while the structure is unchanged. Trials evaluate only the residual vector.

StaticEquilibriumResponse(num_parameters, fixed, initial, options, make_energy, make_response)
returns equilibrium displacement and its direct parameter Jacobian. make_energy(p)
returns the total Energy; make_response(u) returns a parameter-to-gradient Response
at fixed u. Callers guarantee that both describe the same total potential.
Fixed DOFs and values must be parameter-independent.
Each call starts from initial; sensitivities use the exact unshifted Hessian.
eval_jvp and eval_vjp use directional and adjoint solves without forming the
state Jacobian. Parameter products delegate to the returned Response; only its
default product implementations require an explicit parameter Jacobian.
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
