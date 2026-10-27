#pragma once

#include <Eigen/Core>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace litepgo {

// Spatially constant; matrix coefficients flatten by column, independently of Value storage order.
template <class Value> class IdentityField final {
    static_assert(std::is_same_v<Value, typename Value::PlainObject> && std::is_same_v<typename Value::Scalar, double>);

  public:
    class Sample final {
      public:
        const Eigen::Vector3d xi;
        const Value value;
        const Eigen::Matrix<double, Eigen::Dynamic, 3> dvalue_dxi;
        int num_dofs() const {
            return static_cast<int>(value.size());
        }
        Eigen::MatrixXd dvalue_dcoeffs() const {
            return Eigen::MatrixXd::Identity(value.size(), value.size());
        }
        Eigen::MatrixXd dgrad_xi_dcoeffs() const {
            return Eigen::MatrixXd::Zero(3 * value.size(), value.size());
        }

      private:
        friend class IdentityField;
        Sample(Eigen::Vector3d xi, Value value)
            : xi(std::move(xi)), value(std::move(value)),
              dvalue_dxi(Eigen::Matrix<double, Eigen::Dynamic, 3>::Zero(this->value.size(), 3)) {}
    };
    explicit IdentityField(Value coeffs) : coeffs_(std::move(coeffs)) {
        if (coeffs_.size() == 0 || coeffs_.size() > std::numeric_limits<int>::max() || !coeffs_.allFinite())
            throw std::invalid_argument("IdentityField requires nonempty finite values with int-sized DOFs");
    }
    int num_dofs() const {
        return static_cast<int>(coeffs_.size());
    }
    int num_components() const {
        return static_cast<int>(coeffs_.size());
    }
    Value value(const Eigen::Vector3d& xi) const {
        check_xi(xi);
        return coeffs_;
    }
    Eigen::Matrix<double, Eigen::Dynamic, 3> dvalue_dxi(const Eigen::Vector3d& xi) const {
        check_xi(xi);
        return Eigen::Matrix<double, Eigen::Dynamic, 3>::Zero(coeffs_.size(), 3);
    }
    Sample evaluate(const Eigen::Vector3d& xi) const {
        check_xi(xi);
        return Sample(xi, coeffs_);
    }

  private:
    static void check_xi(const Eigen::Vector3d& xi) {
        if (!xi.allFinite())
            throw std::invalid_argument("IdentityField requires finite parameters");
    }
    Value coeffs_;
};

} // namespace litepgo
