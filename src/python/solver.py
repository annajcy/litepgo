"""Nonlinear solvers, mirroring src/core/solver.

Newton checks initial feasibility before convergence, including with all DOFs
fixed. NewtonOptions.max_coordinate_step bounds movement in solver coordinates
before the feasible-step query; its default is infinity. The required convergence policy sees a readonly scalar snapshot before Hessian
assembly and again after computing a direction. Direction fields are None in
the first phase. AbsoluteConvergence runs natively; Python callbacks run at most
twice per iteration and must not mutate solver state.

Gauss-Newton accepts a Response and uses sparse damped normal equations for damped steps.
Jacobian patterns are updated at current iterates; symbolic analysis is reused
while the structure is unchanged. Trials evaluate only the residual vector.

StaticEquilibriumResponse(num_parameters, fixed, initial, options, make_energy, make_response, *, convergence)
returns equilibrium displacement and its direct parameter Jacobian. make_energy(p)
returns the total Energy; make_response(u) returns a parameter-to-gradient Response
at fixed u. Callers guarantee that both describe the same total potential.
The response keeps all callbacks alive and exposes them to Python's cyclic
garbage collector.
Fixed DOFs and values must be parameter-independent.
Each evaluate(p) returns independent state; first value/product use solves from
initial. Its jvp and vjp use directional and adjoint solves without forming the
state Jacobian. The converged state and exact unshifted stiffness factorization
are reused by subsequent derivatives of that evaluation. Parameter products delegate to the returned Response; only its
default product implementations require an explicit parameter Jacobian.
Nonconverged equilibria raise ValueError. See examples/fem/fitting/volume_equilibrium_material_fit.py.

"""

from litepgo._core import solver as _native

ConvergenceState = _native.ConvergenceState
AbsoluteConvergence = _native.AbsoluteConvergence
NewtonOptions = _native.NewtonOptions
NewtonStatus = _native.NewtonStatus
NewtonStep = _native.NewtonStep
NewtonResult = _native.NewtonResult
newton_solve = _native.newton_solve

__all__ = ["ConvergenceState", "AbsoluteConvergence", "NewtonOptions", "NewtonStatus", "NewtonStep", "NewtonResult", "newton_solve"]

GaussNewtonOptions = _native.GaussNewtonOptions
GaussNewtonStatus = _native.GaussNewtonStatus
GaussNewtonResult = _native.GaussNewtonResult
gauss_newton_solve = _native.gauss_newton_solve

__all__ += ["GaussNewtonOptions", "GaussNewtonStatus", "GaussNewtonResult", "gauss_newton_solve"]

StaticEquilibriumResponse = _native.StaticEquilibriumResponse

__all__ += ["StaticEquilibriumResponse"]
