#pragma once

#include "litepgo/common/types.hpp"

namespace litepgo::dof {

// Surface vertex positions as an affine function of the simulation DOFs u:
// p(u) = X + W u, with X the surface's rest positions (vertex-major xyz) and W the
// sparse map from simulation DOFs to surface displacements, 3 * num_vertices() x
// num_dofs(). Selecting vertex DOFs, embedding a surface in elements by
// interpolation weights, and static vertices (zero rows) are all such maps.
//
// An energy B of surface positions, such as contact, acts on u as
// energy::AffineMappedEnergy(B, map.displacement_map(), map.rest_positions()),
// which pulls gradients and Hessians back as W^T g and W^T H W.
class SurfaceKinematics {
  public:
    // rest_positions holds three finite coordinates per surface vertex;
    // displacement_map has finite entries and 3 * num_vertices rows.
    SurfaceKinematics(Vec rest_positions, SpMat displacement_map);

    int num_vertices() const {
        return static_cast<int>(rest_positions_.size() / 3);
    }
    int num_dofs() const {
        return static_cast<int>(displacement_map_.cols());
    }
    const Vec& rest_positions() const {
        return rest_positions_;
    }
    const SpMat& displacement_map() const {
        return displacement_map_;
    }

    // W u and X + W u.
    Vec displacements(const ConstVecRef& u) const;
    Vec positions(const ConstVecRef& u) const;

  private:
    Vec rest_positions_;
    SpMat displacement_map_;
};

} // namespace litepgo::dof
