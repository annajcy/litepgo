"""Sparse linear solvers, mirroring src/cpp/linalg/sparse_solver.

Solvers take scipy.sparse matrices. AccelerateLLT and AccelerateLDLT exist only
in builds with Apple Accelerate, PardisoLLT and PardisoLDLT only in builds with
Intel oneMKL; AutoSparseSolver and AutoIndefiniteSparseSolver are the platform
defaults, as in auto_sparse_solver.hpp.
"""

from litepgo._core import linalg as _native

SparseSolver = _native.SparseSolver
EigenLDLT = _native.EigenLDLT
EigenSparseLU = _native.EigenSparseLU
AutoSparseSolver = _native.AutoSparseSolver
AutoIndefiniteSparseSolver = _native.AutoIndefiniteSparseSolver

__all__ = [
    "SparseSolver", "EigenLDLT", "EigenSparseLU", "AutoSparseSolver", "AutoIndefiniteSparseSolver"
]

if hasattr(_native, "AccelerateLLT"):
    AccelerateLLT = _native.AccelerateLLT
    AccelerateLDLT = _native.AccelerateLDLT
    __all__ += ["AccelerateLLT", "AccelerateLDLT"]

if hasattr(_native, "PardisoLLT"):
    PardisoLLT = _native.PardisoLLT
    PardisoLDLT = _native.PardisoLDLT
    __all__ += ["PardisoLLT", "PardisoLDLT"]
