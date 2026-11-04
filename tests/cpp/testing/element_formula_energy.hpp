#pragma once
#include "litepgo/energy/scalar/mesh_energy.hpp"
#include <limits>

namespace litepgo::testing {
// Test-only mathematical oracles: preserve scalar formulas while exercising the
// public batch protocol. Not an alternate production execution path.
class ElementFormulaEnergy : public energy::CustomMeshEnergy {
    template <class Map, class Formula> static Callback callback(Map map, Formula formula) {
        return [map, formula](std::span<const int> ids, const energy::BatchMatrix& x, energy::BatchMatrix* gradients,
                              energy::BatchMatrix* hessians) {
            const auto n = x.cols();
            Vec values(ids.size());
            if (gradients)
                gradients->resize(ids.size(), n);
            if (hessians)
                hessians->resize(ids.size() * n, n);
            for (Eigen::Index b = 0; b < x.rows(); ++b) {
                auto local = map.gather(ids[b], Vec::Zero(map.num_dofs()));
                local.template reshaped<Eigen::RowMajor>() = x.row(b).transpose();
                Vec g = Vec::Constant(n, std::numeric_limits<double>::quiet_NaN());
                Eigen::MatrixXd H = Eigen::MatrixXd::Constant(n, n, std::numeric_limits<double>::quiet_NaN());
                values[b] = formula(ids[b], local, gradients ? &g : nullptr, hessians ? &H : nullptr);
                if ((gradients && g.size() != n) || (hessians && (H.rows() != n || H.cols() != n)))
                    throw std::logic_error("invalid test formula output");
                if (gradients)
                    gradients->row(b) = g.transpose();
                if (hessians)
                    hessians->middleRows(b * n, n) = H;
            }
            return values;
        };
    }
    template <class Map, class Formula> static Step step(Map map, Formula formula) {
        return [map, formula](std::span<const int> ids, const energy::BatchMatrix& x, const energy::BatchMatrix& dx) {
            Vec steps(ids.size());
            for (Eigen::Index b = 0; b < x.rows(); ++b) {
                auto u = map.gather(ids[b], Vec::Zero(map.num_dofs()));
                auto du = u;
                u.template reshaped<Eigen::RowMajor>() = x.row(b).transpose();
                du.template reshaped<Eigen::RowMajor>() = dx.row(b).transpose();
                steps[b] = formula(ids[b], u, du);
            }
            return steps;
        };
    }

  public:
    template <class Map, class Formula>
    ElementFormulaEnergy(const Map& map, Formula formula)
        : CustomMeshEnergy(map, callback(map, std::move(formula)), 2) {}
    template <class Map, class Formula, class Bound>
    ElementFormulaEnergy(const Map& map, Formula formula, Bound bound)
        : CustomMeshEnergy(map, callback(map, std::move(formula)), 2, step(map, std::move(bound))) {}
};
} // namespace litepgo::testing
