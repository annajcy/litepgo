#pragma once
#include "litepgo/core/types.hpp"
#include "litepgo/mesh/mesh_topology.hpp"
#include <limits>
#include <stdexcept>
#include <vector>
namespace litepgo {
// Owns layout counts; retains no topology or value references.
class ConstantDofMap final {
  public:
    template <int VerticesPerCell>
    ConstantDofMap(const MeshTopology<VerticesPerCell>& topology, int num_dofs)
        : num_elements_(topology.num_cells()), width_(num_dofs) {
        if (width_ <= 0)
            throw std::invalid_argument("ConstantDofMap: invalid DOF count");
    }
    int num_elements() const noexcept {
        return num_elements_;
    }
    int num_dofs() const noexcept {
        return width_;
    }

    std::vector<int> dof_indices(int element) const {
        check_element(element);
        std::vector<int> result(static_cast<std::size_t>(width_));
        for (int i = 0; i < width_; ++i)
            result[static_cast<std::size_t>(i)] = 0 + i;
        return result;
    }
    Eigen::VectorXd gather(int element, const ConstVecRef& global_dof_values) const {
        check_element(element);
        if (global_dof_values.size() != num_dofs())
            throw std::invalid_argument("ConstantDofMap: incorrect global vector size");
        return global_dof_values.segment(0, width_);
    }

  private:
    void check_element(int element) const {
        if (element < 0 || element >= num_elements_)
            throw std::out_of_range("ConstantDofMap: element out of range");
    }
    int num_elements_;
    int width_;
};
} // namespace litepgo
