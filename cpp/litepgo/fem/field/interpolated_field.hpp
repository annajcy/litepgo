#pragma once

#include "litepgo/fem/basis/basis.hpp"

#include <limits>
#include <stdexcept>
#include <unsupported/Eigen/KroneckerProduct>
#include <utility>

namespace litepgo {

// Scalar coefficients flatten by row: m*a+c. Reference derivatives have m rows and three columns.
template <Basis3D B> class InterpolatedField final {
  public:
    static constexpr int num_basis_functions = B::num_basis_functions;
    class Sample final {
      public:
        const Eigen::Vector3d xi;
        const Eigen::VectorXd value;
        const Eigen::Matrix<double, Eigen::Dynamic, 3> dvalue_dxi;
        const typename B::Sample basis_sample;

        int num_dofs() const {
            return num_basis_functions * static_cast<int>(value.size());
        }
        Eigen::MatrixXd dvalue_dcoeffs() const {
            return Eigen::kroneckerProduct(basis_sample.N.transpose(),
                                           Eigen::MatrixXd::Identity(value.size(), value.size()));
        }
        Eigen::MatrixXd dgrad_xi_dcoeffs() const {
            return Eigen::kroneckerProduct(basis_sample.dN_dxi.transpose(),
                                           Eigen::MatrixXd::Identity(value.size(), value.size()));
        }

      private:
        friend class InterpolatedField;
        Sample(Eigen::Vector3d xi, Eigen::VectorXd value, Eigen::Matrix<double, Eigen::Dynamic, 3> derivative,
               typename B::Sample basis_sample)
            : xi(std::move(xi)), value(std::move(value)), dvalue_dxi(std::move(derivative)),
              basis_sample(std::move(basis_sample)) {}
    };

    explicit InterpolatedField(Eigen::MatrixXd coeffs) : coeffs_(std::move(coeffs)) {
        if (coeffs_.rows() != num_basis_functions || coeffs_.cols() == 0 ||
            coeffs_.size() > std::numeric_limits<int>::max() || !coeffs_.allFinite())
            throw std::invalid_argument(
                "InterpolatedField requires compatible finite coefficients with int-sized DOFs");
    }
    int num_dofs() const {
        return static_cast<int>(coeffs_.size());
    }
    int num_components() const {
        return static_cast<int>(coeffs_.cols());
    }

    Eigen::VectorXd value(const Eigen::Vector3d& xi) const {
        const auto N = checked_basis_sample(xi).N;
        Eigen::VectorXd result = coeffs_.transpose() * N;
        if (!N.allFinite() || !result.allFinite())
            throw std::invalid_argument("InterpolatedField::value: non-finite data");
        return result;
    }
    Eigen::Matrix<double, Eigen::Dynamic, 3> dvalue_dxi(const Eigen::Vector3d& xi) const {
        const auto dN_dxi = checked_basis_sample(xi).dN_dxi;
        Eigen::Matrix<double, Eigen::Dynamic, 3> result = coeffs_.transpose() * dN_dxi;
        if (!dN_dxi.allFinite() || !result.allFinite())
            throw std::invalid_argument("InterpolatedField::dvalue_dxi: non-finite data");
        return result;
    }
    Sample evaluate(const Eigen::Vector3d& xi) const {
        auto basis_sample = checked_basis_sample(xi);
        Eigen::VectorXd y = coeffs_.transpose() * basis_sample.N;
        Eigen::Matrix<double, Eigen::Dynamic, 3> derivative = coeffs_.transpose() * basis_sample.dN_dxi;
        if (!basis_sample.N.allFinite() || !basis_sample.dN_dxi.allFinite() || !y.allFinite() ||
            !derivative.allFinite())
            throw std::invalid_argument("InterpolatedField::evaluate: non-finite data");
        return Sample(xi, std::move(y), std::move(derivative), std::move(basis_sample));
    }

  private:
    static typename B::Sample checked_basis_sample(const Eigen::Vector3d& xi) {
        if (!xi.allFinite())
            throw std::invalid_argument("InterpolatedField requires finite parameters");
        return B::evaluate(xi);
    }
    Eigen::MatrixXd coeffs_;
};

} // namespace litepgo
