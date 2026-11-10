#include "litepgo/energy/scalar/quadratic_energy.hpp"

#include "litepgo/linalg/sparse_pattern.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace litepgo::energy {

namespace {

// A, compressed, after checking that it is square and symmetric in pattern and
// values (up to a relative 1e-12, for products like B + B^T computed in floating point).
const SpMat& checked_symmetric(SpMat& A) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("QuadraticEnergy requires a square matrix");
    }
    A.makeCompressed();
    if (!A.coeffs().allFinite())
        throw std::invalid_argument("QuadraticEnergy requires finite matrix entries");
    if (!linalg::is_canonical(A)) {
        throw std::invalid_argument("QuadraticEnergy requires sorted row indices and no duplicate entries");
    }
    const SpMat transposed = SpMat(A.transpose());
    if (!linalg::same_pattern(A, transposed)) {
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

QuadraticEnergy::QuadraticEnergy(SpMat A, Vec b)
    : FixedPatternEnergy(checked_symmetric(A)), A_(std::move(A)), b_(std::move(b)) {
    if (b_.size() == 0)
        b_ = Vec::Zero(A_.rows());
    if (!b_.allFinite())
        throw std::invalid_argument("QuadraticEnergy requires finite linear coefficients");
    if (b_.size() != A_.rows())
        throw std::invalid_argument("QuadraticEnergy: b must have as many entries as A has rows");
}

double QuadraticEnergy::eval_values(const ConstVecRef& x, Vec* gradient, SpMat* hessian) const {
    const Vec Ax = A_ * x;
    if (gradient != nullptr) {
        *gradient = Ax + b_;
    }
    if (hessian != nullptr) {
        std::copy_n(A_.valuePtr(), A_.nonZeros(), hessian->valuePtr());
    }
    return 0.5 * x.dot(Ax) + b_.dot(x);
}

EnergyValues QuadraticEnergy::eval_projected_values(const ConstVecRef& x, EnergyEvaluation&) const {
    std::call_once(projection_once_, [&] {
        bool diagonal = true;
        for (int c = 0; c < A_.outerSize(); ++c)
            for (SpMat::InnerIterator it(A_, c); it; ++it)
                if (it.row() != c && it.value() != 0.)
                    diagonal = false;
        if (diagonal) {
            projected_ = A_;
            projected_.coeffs() = projected_.coeffs().cwiseMax(0.);
        } else
            projected_ = project_hessian(A_);
    });
    EnergyValues result{0., Vec(num_dofs()), projected_};
    result.value = eval_values(x, &*result.gradient, nullptr);
    return result;
}

} // namespace litepgo::energy
