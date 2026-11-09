#pragma once

#include "litepgo/common/types.hpp"

#include <string>
#include <vector>

namespace litepgo::contact {

/// An owned observation of all merged active assembly records at one position vector.
/// Mollified EE retains four support vertices even when its closest distance
/// feature is VV or VE, including records whose mollifier is zero.
struct ContactSnapshot {
    using Positions = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
    using VertexIds = Eigen::Matrix<int, Eigen::Dynamic, 4, Eigen::RowMajor>;

    Positions positions;            ///< All current surface vertices, N x 3, in original vertex numbering.
    std::vector<std::string> types; ///< M record types: VV, VE, VF or EE; rows are not persistent contact IDs.
    VertexIds vertex_ids;           ///< M x 4: vertex/vertex, vertex/edge, vertex/face or edge/edge; pad with -1.
    Vec distances;                  ///< M unsquared distances, without subtracting min_distance.
    Vec weights;                    ///< M sums of area/group weights, excluding stiffness and barrier normalization.
    Vec mollifiers;                 ///< M EE mollifiers (including zero); one for other types. Weights exclude these.
};

} // namespace litepgo::contact
