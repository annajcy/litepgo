"""Energies and their composites, mirroring cpp/litepgo/energy.

Hessians are returned as scipy.sparse.csc_matrix. Before assembling a Hessian of
an energy whose pattern is not fixed, call update_pattern(x) at that x. Values are
always finite; evaluating at an infeasible state raises ValueError.

Numerical entry points are eval_value, eval_gradient, eval_hessian, and
eval_projected_hessian. eval(x, gradient=False, hessian=None) returns the value
and requested derivatives together.
"""

from litepgo._core import energy as _native

HessianKind = _native.HessianKind
Energy = _native.Energy
QuadraticEnergy = _native.QuadraticEnergy
WeightedEnergySum = _native.WeightedEnergySum
AffineMappedEnergy = _native.AffineMappedEnergy

__all__ = ["HessianKind", "Energy", "QuadraticEnergy", "WeightedEnergySum", "AffineMappedEnergy"]
