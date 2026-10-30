#pragma once

#include "litepgo/core/types.hpp"

#include <vector>

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
class SurfaceDofMap {
  public:
    // rest_positions holds three finite coordinates per surface vertex;
    // displacement_map has finite entries and 3 * num_vertices rows.
    SurfaceDofMap(Vec rest_positions, SpMat displacement_map);

    // Surface vertex i displaces with simulation vertex vertices[i], whose position
    // DOFs are the first three of its dofs_per_vertex DOFs (VertexDofMap's layout,
    // e.g. 3 for P1 and Q1, 24 for Hex Hermite's value slot); -1 keeps vertex i at rest.
    SurfaceDofMap(Vec rest_positions, const std::vector<int>& vertices, int num_dofs, int dofs_per_vertex = 3);

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
