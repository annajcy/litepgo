#pragma once
#include "litepgo/core/types.hpp"
#include "litepgo/mesh/mesh_topology.hpp"
#include <limits>
#include <stdexcept>
#include <vector>
namespace litepgo {
// Owns layout counts; retains no topology or value references.
class ElementDofMap final {
  public:
    template <int VerticesPerCell>
    ElementDofMap(const MeshTopology<VerticesPerCell>& topology, int dofs_per_element)
        : num_elements_(topology.num_cells()), width_(dofs_per_element) {
        if (width_ <= 0 || num_elements_ > std::numeric_limits<int>::max() / width_)
            throw std::invalid_argument("ElementDofMap: invalid DOF count");
    }
    int num_elements() const noexcept {
        return num_elements_;
    }
    int num_dofs() const noexcept {
        return num_elements_ * width_;
    }
    int dofs_per_element() const noexcept {
        return width_;
    }
    std::vector<int> dof_indices(int element) const {
        check_element(element);
        std::vector<int> result(static_cast<std::size_t>(width_));
        for (int i = 0; i < width_; ++i)
            result[static_cast<std::size_t>(i)] = element * width_ + i;
        return result;
    }
    Eigen::VectorXd gather(int element, const ConstVecRef& global_dof_values) const {
        check_element(element);
        if (global_dof_values.size() != num_dofs())
            throw std::invalid_argument("ElementDofMap: incorrect global vector size");
        return global_dof_values.segment(element * width_, width_);
    }

  private:
    void check_element(int element) const {
        if (element < 0 || element >= num_elements_)
            throw std::out_of_range("ElementDofMap: element out of range");
    }
    int num_elements_;
    int width_;
};
} // namespace litepgo
