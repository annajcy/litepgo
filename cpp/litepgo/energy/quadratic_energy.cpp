#include "litepgo/energy/quadratic_energy.hpp"

#include "litepgo/linalg/sparse_pattern.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace litepgo {

namespace {

// A, compressed, after checking that it is square and symmetric in pattern and
// values (up to a relative 1e-12, for products like B + B^T computed in floating point).
SpMat checked_symmetric(SpMat A) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("QuadraticEnergy requires a square matrix");
    }
    A.makeCompressed();
    if (!is_canonical(A)) {
        throw std::invalid_argument("QuadraticEnergy requires sorted row indices and no duplicate entries");
    }
    const SpMat transposed = SpMat(A.transpose());
    if (!same_pattern(A, transposed)) {
        throw std::invalid_argument("QuadraticEnergy requires a symmetric sparsity pattern");
    }
    double scale = 0.0;
    double asymmetry = 0.0;
    for (Eigen::Index k = 0; k < A.nonZeros(); ++k) {
        scale = std::max(scale, std::abs(A.valuePtr()[k]));
        asymmetry = std::max(asymmetry, std::abs(A.valuePtr()[k] - transposed.valuePtr()[k]));
    }
    if (asymmetry > 1e-12 * scale) {
        throw std::invalid_argument("QuadraticEnergy requires a symmetric matrix");
    }
    return A;
}

} // namespace

QuadraticEnergy::Setup QuadraticEnergy::checked_setup(SpMat A, Vec b) {
    A = checked_symmetric(std::move(A));
    if (b.size() == 0) {
        b = Vec::Zero(A.rows());
    }
    if (b.size() != A.rows()) {
        throw std::invalid_argument("QuadraticEnergy: b must have as many entries as A has rows");
    }
    return {std::move(A), std::move(b)};
}

QuadraticEnergy::QuadraticEnergy(SpMat A, Vec b) : QuadraticEnergy(checked_setup(std::move(A), std::move(b))) {}

QuadraticEnergy::QuadraticEnergy(Setup setup)
    : FixedPatternEnergy(static_cast<int>(setup.A.rows()), setup.A), A_(std::move(setup.A)), b_(std::move(setup.b)) {}

double QuadraticEnergy::eval_values(const ConstVecRef& x, Vec* gradient, SpMat* hessian, HessianKind) const {
    const Vec Ax = A_ * x;
    if (gradient != nullptr) {
        *gradient = Ax + b_;
    }
    if (hessian != nullptr) {
        std::copy_n(A_.valuePtr(), A_.nonZeros(), hessian->valuePtr());
    }
    return 0.5 * x.dot(Ax) + b_.dot(x);
}

} // namespace litepgo
