#pragma once
#include "litepgo/dof/vertex_dof_map.hpp"
#include "litepgo/fem/element_locator.hpp"
#include "litepgo/fem/kinematics/basis_dof_transform.hpp"

namespace litepgo::fem {
// X_fields and the DOF map use the same element order as the embedding.
template <class UBasis, class XField>
SpMat build_point_displacement_map(const PointEmbedding& embedding, const dof::VertexDofMap& u_dof_map,
                                   const std::vector<XField>& X_fields) {
    constexpr int width = BasisDofTransform<UBasis>::dofs_per_vertex;
    constexpr int local_size = 3 * UBasis::num_basis_functions;
    if (embedding.element_ids.size() != embedding.xi.rows() || !embedding.xi.allFinite() ||
        embedding.xi.rows() > std::numeric_limits<int>::max() / 3 ||
        X_fields.size() != static_cast<std::size_t>(u_dof_map.num_elements()) || u_dof_map.dofs_per_vertex() != width)
        throw std::invalid_argument("build_point_displacement_map: incompatible embedding, fields or DOF layout");
    std::vector<BasisDofTransform<UBasis>> transforms;
    std::vector<std::vector<int>> columns;
    for (int e = 0; e < u_dof_map.num_elements(); ++e) {
        if (X_fields[e].num_components() != 3)
            throw std::invalid_argument("build_point_displacement_map: X must have three components");
        columns.push_back(u_dof_map.dof_indices(e));
        if (columns.back().size() != local_size)
            throw std::invalid_argument("build_point_displacement_map: element topology does not match basis");
        transforms.emplace_back(X_fields[e]);
    }
    std::vector<Triplet> entries;
    for (Eigen::Index i = 0; i < embedding.xi.rows(); ++i) {
        const int e = embedding.element_ids[i];
        if (e < 0 || e >= u_dof_map.num_elements())
            throw std::out_of_range("build_point_displacement_map: invalid element ID");
        const Eigen::Vector3d xi = embedding.xi.row(i).transpose();
        const bool inside =
            (xi.array() >= -1e-10).all() &&
            (std::is_same_v<UBasis, TetP1Basis> ? xi.sum() <= 1 + 1e-10 : (xi.array() <= 1 + 1e-10).all());
        if (!inside)
            throw std::invalid_argument("build_point_displacement_map: xi outside reference element");
        const auto N = UBasis::eval_N(xi);
        for (int c = 0; c < 3; ++c) {
            Vec row = Vec::Zero(local_size);
            for (int a = 0; a < UBasis::num_basis_functions; ++a)
                row[3 * a + c] = N[a];
            // A row of the interpolation Jacobian pulls back exactly like a gradient.
            transforms[e].pullback(&row, nullptr);
            if (!row.allFinite())
                throw std::domain_error("build_point_displacement_map: nonfinite interpolation weights");
            for (int j = 0; j < local_size; ++j)
                if (row[j] != 0)
                    entries.emplace_back(3 * static_cast<int>(i) + c, columns[e][j], row[j]);
        }
    }
    SpMat W(3 * embedding.xi.rows(), u_dof_map.num_dofs());
    W.setFromTriplets(entries.begin(), entries.end());
    return W;
}
} // namespace litepgo::fem
