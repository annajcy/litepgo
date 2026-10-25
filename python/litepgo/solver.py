"""Nonlinear solvers, mirroring cpp/litepgo/solver."""

from litepgo._core import solver as _native

NewtonOptions = _native.NewtonOptions
NewtonStatus = _native.NewtonStatus
NewtonStep = _native.NewtonStep
NewtonResult = _native.NewtonResult
newton_solve = _native.newton_solve

__all__ = ["NewtonOptions", "NewtonStatus", "NewtonStep", "NewtonResult", "newton_solve"]
