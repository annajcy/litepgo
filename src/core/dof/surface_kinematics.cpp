#include "litepgo/dof/surface_kinematics.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace litepgo::dof {

SurfaceKinematics::SurfaceKinematics(Vec rest_positions, SpMat displacement_map)
    : rest_positions_(std::move(rest_positions)), displacement_map_(std::move(displacement_map)) {
    if (rest_positions_.size() % 3 != 0 || rest_positions_.size() / 3 > std::numeric_limits<int>::max() / 3 ||
        !rest_positions_.allFinite()) {
        throw std::invalid_argument("SurfaceKinematics requires three finite rest coordinates per surface vertex");
    }
    if (displacement_map_.rows() != rest_positions_.size()) {
        throw std::invalid_argument("SurfaceKinematics: the displacement map must have 3 * num_vertices rows");
    }
    displacement_map_.makeCompressed();
    if (!Eigen::Map<const Vec>(displacement_map_.valuePtr(), displacement_map_.nonZeros()).allFinite()) {
        throw std::invalid_argument("SurfaceKinematics requires a finite displacement map");
    }
}

Vec SurfaceKinematics::displacements(const ConstVecRef& u) const {
    if (u.size() != num_dofs()) {
        throw std::invalid_argument("SurfaceKinematics: u must have num_dofs entries");
    }
    return displacement_map_ * u;
}

Vec SurfaceKinematics::positions(const ConstVecRef& u) const {
    return rest_positions_ + displacements(u);
}

} // namespace litepgo::dof
