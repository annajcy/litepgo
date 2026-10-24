#pragma once

// The platform's default sparse solvers, chosen at compile time.

#if defined(LITEPGO_HAS_ACCELERATE)
#include "litepgo/linalg/sparse_solver/accelerate_ldlt.hpp"
#include "litepgo/linalg/sparse_solver/accelerate_llt.hpp"
#else
#include "litepgo/linalg/sparse_solver/eigen_ldlt.hpp"
#include "litepgo/linalg/sparse_solver/eigen_sparse_lu.hpp"
#endif

namespace litepgo {

#if defined(LITEPGO_HAS_ACCELERATE)
using AutoSparseSolver = AccelerateLLT;
using AutoIndefiniteSparseSolver = AccelerateLDLT;
#else
using AutoSparseSolver = EigenLDLT;
using AutoIndefiniteSparseSolver = EigenSparseLU;
#endif

} // namespace litepgo
