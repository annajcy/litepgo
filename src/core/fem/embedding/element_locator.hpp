#pragma once
#include "litepgo/fem/basis/3d/hex_q1_basis.hpp"
#include "litepgo/fem/basis/3d/tet_p1_basis.hpp"
#include "litepgo/fem/response/interpolated_response.hpp"
#include "litepgo/spatial/spatial_hash.hpp"
#include <Eigen/LU>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>

namespace litepgo::fem {
struct PointEmbedding {
    Eigen::VectorXi element_ids;
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> xi;
};

// Geometry basis, independent of the displacement basis. Owns a rest-geometry snapshot.
// shortcut: supports P1/Q1 geometry only; curved higher-order fields need conservative bounds.
template <class B>
    requires(std::is_same_v<B, TetP1Basis> || std::is_same_v<B, HexQ1Basis>)
class ElementLocator {
    std::vector<std::shared_ptr<energy::Response>> geometry_;
    std::vector<double> tolerances_;
    spatial::SpatialHash index_;

    static std::vector<std::shared_ptr<energy::Response>>
    snapshot(const std::vector<std::shared_ptr<energy::Response>>& geometry) {
        std::vector<std::shared_ptr<energy::Response>> result;
        for (const auto& response : geometry) {
            // Corner bounds are conservative only for the declared P1/Q1 interpolation.
            if (!response || !dynamic_cast<InterpolatedResponse<B>*>(response.get()) || response->num_outputs() != 3)
                throw std::invalid_argument(
                    "ElementLocator: expected three-component InterpolatedResponse of geometry basis");
            Eigen::MatrixXd coefficients(B::num_basis_functions, 3);
            for (int a = 0; a < B::num_basis_functions; ++a)
                coefficients.row(a) = response->eval_value(B::coefficient_info(a).point).transpose();
            result.push_back(std::make_shared<InterpolatedResponse<B>>(coefficients));
        }
        return result;
    }
    std::vector<spatial::AABB> bounds() {
        std::vector<spatial::AABB> result;
        for (const auto& field : geometry_) {
            if (field->num_outputs() != 3)
                throw std::invalid_argument("ElementLocator: X fields must have three components");
            Eigen::Vector3d lo = field->eval_value(B::coefficient_info(0).point), hi = lo;
            for (int i = 1; i < B::num_basis_functions; ++i) {
                const Eigen::Vector3d p = field->eval_value(B::coefficient_info(i).point);
                lo = lo.cwiseMin(p);
                hi = hi.cwiseMax(p);
            }
            const double scale = (hi - lo).maxCoeff();
            const double magnitude = std::max(lo.cwiseAbs().maxCoeff(), hi.cwiseAbs().maxCoeff());
            const double tolerance = 1e-10 * scale + 32 * std::numeric_limits<double>::epsilon() * magnitude;
            if (!std::isfinite(scale) || scale <= 0 || !std::isfinite(tolerance))
                throw std::invalid_argument("ElementLocator: invalid geometry extent");
            tolerances_.push_back(tolerance);
            result.push_back({lo.array() - tolerance, hi.array() + tolerance});
        }
        return result;
    }
    static bool inside(const Eigen::Vector3d& xi) {
        if ((xi.array() < -1e-10).any())
            return false;
        if constexpr (std::is_same_v<B, TetP1Basis>)
            return xi.sum() <= 1 + 1e-10;
        else
            return (xi.array() <= 1 + 1e-10).all();
    }
    bool invert(std::size_t e, const Eigen::Vector3d& p, Eigen::Vector3d& xi) const {
        const auto& field = geometry_[e];
        xi.setConstant(std::is_same_v<B, TetP1Basis> ? 0.25 : 0.5);
        for (int iteration = 0; iteration < 40; ++iteration) {
            const Eigen::Vector3d residual = field->eval_value(xi) - p;
            const Eigen::Matrix3d J = field->eval_jacobian(xi);
            Eigen::FullPivLU<Eigen::Matrix3d> lu(J);
            if (!J.allFinite() || !lu.isInvertible())
                return false;
            if (residual.lpNorm<Eigen::Infinity>() <= tolerances_[e])
                return inside(xi);
            const Eigen::Vector3d step = lu.solve(residual);
            if (!step.allFinite())
                return false;
            bool accepted = false;
            for (double alpha = 1; alpha >= 1.0 / 1024; alpha *= 0.5) {
                const Eigen::Vector3d next = xi - alpha * step;
                if (!next.allFinite())
                    continue;
                const Eigen::Vector3d next_residual = field->eval_value(next) - p;
                if (next_residual.norm() < residual.norm()) {
                    xi = next;
                    accepted = true;
                    break;
                }
            }
            if (!accepted)
                return false;
        }
        return false;
    }

  public:
    explicit ElementLocator(const std::vector<std::shared_ptr<energy::Response>>& geometry)
        : geometry_(snapshot(geometry)), index_(bounds()) {}

    // Boundary points are accepted within tolerance; lowest valid element ID wins.
    // Failure throws with the point index; no extrapolation or nearest-element fallback.
    PointEmbedding locate(const Eigen::Ref<const Eigen::MatrixXd>& points) const {
        if (points.cols() != 3 || !points.allFinite())
            throw std::invalid_argument("ElementLocator: points must be finite (n, 3)");
        PointEmbedding result{Eigen::VectorXi(points.rows()), decltype(PointEmbedding::xi)(points.rows(), 3)};
        for (Eigen::Index i = 0; i < points.rows(); ++i) {
            const Eigen::Vector3d p = points.row(i).transpose();
            auto candidates = index_.query({p, p});
            std::sort(candidates.begin(), candidates.end());
            bool found = false;
            for (auto e : candidates) {
                Eigen::Vector3d xi;
                if (invert(e, p, xi)) {
                    result.element_ids[i] = static_cast<int>(e);
                    result.xi.row(i) = xi.transpose();
                    found = true;
                    break;
                }
            }
            if (!found)
                throw std::runtime_error("ElementLocator: could not locate point " + std::to_string(i) +
                                         " (outside, degenerate geometry, or inverse solve failed)");
        }
        return result;
    }
};
} // namespace litepgo::fem
