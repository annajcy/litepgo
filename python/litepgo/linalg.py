"""Sparse linear solvers and DOF reduction, mirroring cpp/litepgo/linalg.

Solvers and DofReduction take scipy.sparse matrices. AccelerateLLT and AccelerateLDLT exist only
in builds with Apple Accelerate; AutoSparseSolver and AutoIndefiniteSparseSolver
are the platform defaults, as in auto_sparse_solver.h.
"""

from litepgo._core import linalg as _native

SparseSolver = _native.SparseSolver
EigenLDLT = _native.EigenLDLT
EigenSparseLU = _native.EigenSparseLU
AutoSparseSolver = _native.AutoSparseSolver
AutoIndefiniteSparseSolver = _native.AutoIndefiniteSparseSolver
DofReduction = _native.DofReduction

__all__ = [
    "SparseSolver", "EigenLDLT", "EigenSparseLU", "AutoSparseSolver", "AutoIndefiniteSparseSolver", "DofReduction"
]

if hasattr(_native, "AccelerateLLT"):
    AccelerateLLT = _native.AccelerateLLT
    AccelerateLDLT = _native.AccelerateLDLT
    __all__ += ["AccelerateLLT", "AccelerateLDLT"]
