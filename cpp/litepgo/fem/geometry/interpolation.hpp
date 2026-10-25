#pragma once

#include "litepgo/fem/basis/basis.hpp"

#include <limits>
#include <stdexcept>
#include <unsupported/Eigen/KroneckerProduct>
#include <utility>

namespace litepgo {

// Rows are basis coefficients, columns are components; DOFs flatten by row.
// Gradients are with respect to xi; their output vectors flatten by column.
template <Basis3D B> class Interpolation final {
  public:
    static constexpr int num_basis_functions = B::num_basis_functions;

    explicit Interpolation(Eigen::MatrixXd coefficients) : coefficients_(std::move(coefficients)) {
        if (coefficients_.rows() != num_basis_functions || coefficients_.cols() == 0 ||
            coefficients_.size() > std::numeric_limits<int>::max() || !coefficients_.allFinite())
            throw std::invalid_argument("Interpolation requires compatible finite coefficients with int-sized DOFs");
    }

    int num_dofs() const {
        return static_cast<int>(coefficients_.size());
    }
    int num_components() const {
        return static_cast<int>(coefficients_.cols());
    }

    Eigen::VectorXd value(const Eigen::Vector3d& xi) const {
        const auto values = basis_sample(xi).values;
        Eigen::VectorXd result = coefficients_.transpose() * values;
        if (!values.allFinite() || !result.allFinite())
            throw std::invalid_argument("Interpolation::value: non-finite data");
        return result;
    }

    Eigen::Matrix<double, Eigen::Dynamic, 3> gradient(const Eigen::Vector3d& xi) const {
        const auto gradients = basis_sample(xi).gradients;
        Eigen::Matrix<double, Eigen::Dynamic, 3> result = coefficients_.transpose() * gradients;
        if (!gradients.allFinite() || !result.allFinite())
            throw std::invalid_argument("Interpolation::gradient: non-finite data");
        return result;
    }

    Eigen::MatrixXd value_matrix(const Eigen::Vector3d& xi) const {
        const auto values = basis_sample(xi).values;
        if (!values.allFinite())
            throw std::invalid_argument("Interpolation::value_matrix: non-finite basis values");
        return Eigen::kroneckerProduct(values.transpose(),
                                       Eigen::MatrixXd::Identity(num_components(), num_components()));
    }

    Eigen::MatrixXd gradient_matrix(const Eigen::Vector3d& xi) const {
        const auto gradients = basis_sample(xi).gradients;
        if (!gradients.allFinite())
            throw std::invalid_argument("Interpolation::gradient_matrix: non-finite basis gradients");
        return Eigen::kroneckerProduct(gradients.transpose(),
                                       Eigen::MatrixXd::Identity(num_components(), num_components()));
    }

  private:
    static typename B::Sample basis_sample(const Eigen::Vector3d& xi) {
        if (!xi.allFinite())
            throw std::invalid_argument("Interpolation requires finite parameters");
        return B::evaluate(xi);
    }

    Eigen::MatrixXd coefficients_;
};

} // namespace litepgo
