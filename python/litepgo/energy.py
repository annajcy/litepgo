"""Energies and their composites, mirroring cpp/litepgo/energy.

Hessians are returned as scipy.sparse.csc_matrix. Before assembling a Hessian of
an energy whose pattern is not fixed, call update_pattern(x) at that x. Values are
always finite; evaluating at an infeasible state raises InfeasibleStateError.
"""

from litepgo._core import energy as _native

HessianKind = _native.HessianKind
InfeasibleStateError = _native.InfeasibleStateError
Energy = _native.Energy
QuadraticEnergy = _native.QuadraticEnergy
WeightedEnergySum = _native.WeightedEnergySum
AffineMappedEnergy = _native.AffineMappedEnergy

__all__ = ["HessianKind", "InfeasibleStateError", "Energy", "QuadraticEnergy", "WeightedEnergySum", "AffineMappedEnergy"]
