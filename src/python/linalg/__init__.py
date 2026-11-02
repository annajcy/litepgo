"""Linear algebra: sparse solvers."""

from . import sparse_solver
from .sparse_solver import *

__all__ = ["sparse_solver", *sparse_solver.__all__]
