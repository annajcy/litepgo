#pragma once

#include "litepgo/fem/dof/constant_dof_data.hpp"
#include "litepgo/mesh/mesh_topology.hpp"

#include <array>
#include <stdexcept>

namespace litepgo {

// Captures the cell count; retains no topology reference.
// Indices address this field's vector, in scalar slot order.
class ConstantDofMap final {
  public:
    template <int VerticesPerCell>
    explicit ConstantDofMap(const MeshTopology<VerticesPerCell>& topology) : num_elements_(topology.num_cells()) {}

    // Returns an independent value snapshot.
    template <int NumDofs>
    Eigen::Matrix<double, NumDofs, 1> gather(int element, const ConstantDofData<NumDofs>& data) const {
        check_element(element);
        return data.values();
    }

    template <int NumDofs> std::array<int, NumDofs> dof_indices(int element, const ConstantDofData<NumDofs>&) const {
        check_element(element);
        std::array<int, NumDofs> result;
        for (int i = 0; i < NumDofs; ++i)
            result[static_cast<std::size_t>(i)] = i;
        return result;
    }

  private:
    void check_element(int element) const {
        if (element < 0 || element >= num_elements_)
            throw std::out_of_range("ConstantDofMap: element is out of range");
    }
    int num_elements_;
};

} // namespace litepgo
