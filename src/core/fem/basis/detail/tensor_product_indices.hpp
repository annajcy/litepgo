#pragma once

#include "litepgo/fem/basis/basis.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace litepgo::fem::detail {

// One shared row order for values, derivatives and coefficient descriptions.
template <Basis1D B, int Dim> const auto& tensor_product_indices() {
    static const auto indices = [] {
        constexpr int n = B::num_basis_functions;
        constexpr int count = Dim == 2 ? n * n : n * n * n;
        std::array<CoefficientInfo<1>, n> info;
        double first_point = 0;
        for (int i = 0; i < n; ++i) {
            info[i] = B::coefficient_info(i);
            if (!std::isfinite(info[i].point) || info[i].derivative_order[0] < 0)
                throw std::invalid_argument("Tensor product requires finite points and nonnegative derivative orders");
            if (i == 0 || info[i].point < first_point)
                first_point = info[i].point;
        }
        std::array<std::array<int, Dim>, count> rows{};
        for (int row = 0; row < count; ++row) {
            int index = row;
            for (int d = Dim - 1; d >= 0; --d) {
                rows[row][d] = index % n;
                index /= n;
            }
        }
        std::sort(rows.begin(), rows.end(), [&](const auto& a, const auto& b) {
            // Traverse reference nodes by layers, with the first axis reversed on upper rows.
            for (int d = Dim - 1; d > 0; --d)
                if (info[a[d]].point != info[b[d]].point)
                    return info[a[d]].point < info[b[d]].point;
            if (info[a[0]].point != info[b[0]].point)
                return info[a[1]].point == first_point ? info[a[0]].point < info[b[0]].point
                                                       : info[a[0]].point > info[b[0]].point;
            long long degree_a = 0, degree_b = 0;
            for (int d = 0; d < Dim; ++d) {
                degree_a += info[a[d]].derivative_order[0];
                degree_b += info[b[d]].derivative_order[0];
            }
            if (degree_a != degree_b)
                return degree_a < degree_b;
            for (int d = 0; d < Dim; ++d)
                if (info[a[d]].derivative_order != info[b[d]].derivative_order)
                    return info[a[d]].derivative_order > info[b[d]].derivative_order;
            return a < b;
        });
        return rows;
    }();
    return indices;
}

} // namespace litepgo::fem::detail
