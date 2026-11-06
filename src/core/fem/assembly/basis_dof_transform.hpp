#pragma once

#include "litepgo/common/types.hpp"
#include "litepgo/energy/response/response.hpp"
#include "litepgo/fem/basis/3d/hex_hermite_basis.hpp"
#include "litepgo/fem/basis/3d/hex_q1_basis.hpp"
#include "litepgo/fem/basis/3d/tet_p1_basis.hpp"
#include <array>
#include <stdexcept>

namespace litepgo::fem {

// No fallback: every supported basis must specify its physical DOF convention.
template <class Basis> class BasisDofTransform;

namespace detail {
template <int Nodes> class NodalDofTransform {
  public:
    static constexpr int dofs_per_vertex = 3;
    explicit NodalDofTransform(energy::Response& geometry) {
        if (geometry.num_inputs() != 3 || geometry.num_outputs() != 3)
            throw std::invalid_argument("BasisDofTransform: geometry must map R3 to R3");
    }
    Eigen::MatrixXd to_coefficients(Eigen::MatrixXd local) const {
        if (local.rows() != Nodes || local.cols() != 3 || !local.allFinite())
            throw std::invalid_argument("BasisDofTransform: invalid nodal DOFs");
        return local;
    }
    void pullback(Eigen::VectorXd* gradient, Eigen::MatrixXd* hessian) const {
        pullback_impl(gradient, hessian);
    }
    void pullback_into(StridedVecRef* gradient, StridedMatRef* hessian) const {
        pullback_impl(gradient, hessian);
    }
    // Force and parameter Jacobian: transform output rows only (T^T g, T^T J).
    void pullback_response_into(StridedVecRef value, StridedMatRef* jacobian) const {
        if (value.size() != 3 * Nodes || (jacobian && jacobian->rows() != 3 * Nodes))
            throw std::invalid_argument("BasisDofTransform: invalid response dimensions");
    }

  private:
    template <class G, class H> void pullback_impl(G* gradient, H* hessian) const {
        if ((gradient && gradient->size() != 3 * Nodes) ||
            (hessian && (hessian->rows() != 3 * Nodes || hessian->cols() != 3 * Nodes)))
            throw std::invalid_argument("BasisDofTransform: invalid derivative dimensions");
    }
};
} // namespace detail

template <> class BasisDofTransform<TetP1Basis> : public detail::NodalDofTransform<4> {
  public:
    using NodalDofTransform::NodalDofTransform;
};
template <> class BasisDofTransform<HexQ1Basis> : public detail::NodalDofTransform<8> {
  public:
    using NodalDofTransform::NodalDofTransform;
};

template <> class BasisDofTransform<HexHermiteBasis> {
  public:
    // Per vertex: u, u_X, u_Y, u_Z, u_XY, u_XZ, u_YZ, u_XYZ, each a 3-vector.
    static constexpr int dofs_per_vertex = 24;
    // X must represent a positive axis-aligned affine box throughout the reference domain.
    // shortcut: only corners are checked; curved geometry needs a different mapping.
    explicit BasisDofTransform(energy::Response& X) {
        Eigen::MatrixXd nodes(8, 3);
        if ((X.num_inputs() != 3 || X.num_outputs() != 3))
            throw std::invalid_argument("Hermite DOFs require three reference components");
        for (int v = 0; v < 8; ++v) {
            const auto position = X.eval_value(HexQ1Basis::coefficient_info(v).point);
            if (position.size() != 3 || !position.allFinite())
                throw std::invalid_argument("Hermite DOFs require finite three-component reference values");
            nodes.row(v) = position.transpose();
        }
        const Eigen::Vector3d origin = nodes.row(0).transpose();
        const Eigen::Vector3d lengths(nodes(1, 0) - origin.x(), nodes(3, 1) - origin.y(), nodes(4, 2) - origin.z());
        if (!lengths.allFinite() || (lengths.array() <= 0).any())
            throw std::invalid_argument("Hermite DOF transform requires positive axis-aligned box dimensions");
        for (int v = 0; v < 8; ++v) {
            const Eigen::Vector3d normalized = (nodes.row(v).transpose() - origin).cwiseQuotient(lengths);
            if (!normalized.allFinite() || !(normalized - HexHermiteBasis::coefficient_info(8 * v).point).isZero(1e-12))
                throw std::invalid_argument("Hermite DOF transform requires axis-aligned boxes in Hex corner order");
        }
        const auto [hx, hy, hz] = std::array{lengths.x(), lengths.y(), lengths.z()};
        const Eigen::Matrix<double, 8, 1> slots{1, hx, hy, hz, hx * hy, hx * hz, hy * hz, hx * hy * hz};
        scales_.resize(3 * HexHermiteBasis::num_basis_functions);
        for (int v = 0; v < 8; ++v)
            for (int slot = 0; slot < 8; ++slot)
                scales_.segment<3>(3 * (8 * v + slot)).setConstant(slots[slot]);
        if (!scales_.allFinite() || scales_.minCoeff() <= 0)
            throw std::invalid_argument("Hermite DOF transform requires finite positive DOF scales");
    }
    Eigen::MatrixXd to_coefficients(const Eigen::MatrixXd& local) const {
        if (local.rows() != 8 || local.cols() != 24 || !local.allFinite())
            throw std::invalid_argument("Hex Hermite requires finite 8x24 physical DOFs");
        Eigen::MatrixXd u_local = local.reshaped<Eigen::RowMajor>(64, 3);
        for (int row = 0; row < 64; ++row)
            u_local.row(row) *= scales_[3 * row];
        if (!u_local.allFinite())
            throw std::domain_error("Hex Hermite: non-finite reference coefficients");
        return u_local;
    }
    void pullback(Eigen::VectorXd* gradient, Eigen::MatrixXd* hessian) const {
        pullback_impl(gradient, hessian);
    }
    void pullback_into(StridedVecRef* gradient, StridedMatRef* hessian) const {
        pullback_impl(gradient, hessian);
    }
    // The input parameters are unchanged, so the Jacobian receives no right transform.
    void pullback_response_into(StridedVecRef value, StridedMatRef* jacobian) const {
        if (value.size() != scales_.size() || (jacobian && jacobian->rows() != scales_.size()))
            throw std::invalid_argument("Hermite DOFs: invalid response dimensions");
        value.array() *= scales_.array();
        if (jacobian)
            jacobian->array().colwise() *= scales_.array();
        if (!value.allFinite() || (jacobian && !jacobian->allFinite()))
            throw std::domain_error("Hermite DOFs: non-finite physical response");
    }

  private:
    template <class G, class H> void pullback_impl(G* gradient, H* hessian) const {
        if ((gradient && gradient->size() != scales_.size()) ||
            (hessian && (hessian->rows() != scales_.size() || hessian->cols() != scales_.size())))
            throw std::invalid_argument("Hermite DOFs: invalid derivative dimensions");
        if (gradient)
            gradient->array() *= scales_.array();
        const auto scale_hessian = [&](auto& matrix) {
            matrix.array().colwise() *= scales_.array();
            matrix.array().rowwise() *= scales_.transpose().array();
            if (!matrix.allFinite())
                throw std::domain_error("Hermite DOFs: non-finite physical derivatives");
        };
        if (hessian) {
            if (hessian->colStride() == 1) {
                using RowMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
                Eigen::Map<RowMatrix, 0, Eigen::OuterStride<>> rows(hessian->data(), hessian->rows(), hessian->cols(),
                                                                    Eigen::OuterStride<>(hessian->rowStride()));
                scale_hessian(rows);
            } else {
                scale_hessian(*hessian);
            }
        }
        if (gradient && !gradient->allFinite())
            throw std::domain_error("Hermite DOFs: non-finite physical derivatives");
    }

    Eigen::VectorXd scales_;
};

} // namespace litepgo::fem
