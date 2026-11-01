#pragma once
#include "litepgo/response/mesh_response.hpp"
#include <limits>

namespace litepgo::testing {
// Test-only scalar oracle evaluated for each row of the public batch protocol.
template <class Input, class Output, class Formula> auto formula_response(Input input, Output output, Formula formula) {
    return response::MeshResponse(
        input, output,
        [input, output, formula](std::span<const int> ids, const response::BatchMatrix& x, response::BatchMatrix* J) {
            const Eigen::Index no = output.dof_indices(ids.front()).size(), ni = x.cols();
            response::BatchMatrix values(ids.size(), no);
            if (J)
                J->resize(ids.size() * no, ni);
            for (Eigen::Index b = 0; b < x.rows(); ++b) {
                auto local = input.gather(ids[b], Vec::Zero(input.num_dofs()));
                local.template reshaped<Eigen::RowMajor>() = x.row(b).transpose();
                Eigen::MatrixXd derivative =
                    Eigen::MatrixXd::Constant(no, ni, std::numeric_limits<double>::quiet_NaN());
                const Vec value = formula(ids[b], local, J ? &derivative : nullptr);
                if (value.size() != no || (J && (derivative.rows() != no || derivative.cols() != ni)))
                    throw std::logic_error("invalid local test formula output");
                values.row(b) = value.transpose();
                if (J)
                    J->middleRows(b * no, no) = derivative;
            }
            return values;
        },
        2);
}
} // namespace litepgo::testing
