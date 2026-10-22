#pragma once

// The platform's default sparse solvers, chosen at compile time.

#if defined(LITEPGO_HAS_ACCELERATE)
#include "litepgo/linalg/accelerate_ldlt.h"
#include "litepgo/linalg/accelerate_llt.h"
#else
#include "litepgo/linalg/eigen_ldlt.h"
#include "litepgo/linalg/eigen_sparse_lu.h"
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
