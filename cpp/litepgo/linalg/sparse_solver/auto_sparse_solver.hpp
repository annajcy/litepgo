#pragma once

// The platform's default sparse solvers, chosen at compile time.

#if defined(LITEPGO_HAS_ACCELERATE)
#include "litepgo/linalg/sparse_solver/accelerate_ldlt.hpp"
#include "litepgo/linalg/sparse_solver/accelerate_llt.hpp"
#elif defined(LITEPGO_HAS_MKL)
#include "litepgo/linalg/sparse_solver/pardiso_ldlt.hpp"
#include "litepgo/linalg/sparse_solver/pardiso_llt.hpp"
#else
#include "litepgo/linalg/sparse_solver/eigen_ldlt.hpp"
#include "litepgo/linalg/sparse_solver/eigen_sparse_lu.hpp"
#endif

namespace litepgo::linalg {

#if defined(LITEPGO_HAS_ACCELERATE)
using AutoSparseSolver = AccelerateLLT;
using AutoIndefiniteSparseSolver = AccelerateLDLT;
#elif defined(LITEPGO_HAS_MKL)
using AutoSparseSolver = PardisoLLT;
using AutoIndefiniteSparseSolver = PardisoLDLT;
#else
using AutoSparseSolver = EigenLDLT;
using AutoIndefiniteSparseSolver = EigenSparseLU;
#endif

} // namespace litepgo::linalg
