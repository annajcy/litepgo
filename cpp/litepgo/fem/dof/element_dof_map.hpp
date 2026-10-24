#pragma once

#include "litepgo/fem/dof/element_dof_data.hpp"
#include "litepgo/mesh/mesh_topology.hpp"

#include <array>
#include <stdexcept>

namespace litepgo {

// Captures the cell count; retains no topology reference.
// Indices address this field's vector, in scalar slot order.
class ElementDofMap final {
  public:
    template <int VerticesPerCell>
    explicit ElementDofMap(const MeshTopology<VerticesPerCell>& topology) : num_elements_(topology.num_cells()) {}

    // Returns an independent value snapshot.
    template <int DofsPerElement>
    Eigen::Matrix<double, DofsPerElement, 1> gather(int element, const ElementDofData<DofsPerElement>& data) const {
        check(element, data);
        return data.values().template segment<DofsPerElement>(element * DofsPerElement);
    }

    template <int DofsPerElement>
    std::array<int, DofsPerElement> dof_indices(int element, const ElementDofData<DofsPerElement>& data) const {
        check(element, data);
        std::array<int, DofsPerElement> result;
        for (int i = 0; i < DofsPerElement; ++i)
            result[static_cast<std::size_t>(i)] = element * DofsPerElement + i;
        return result;
    }

  private:
    template <int DofsPerElement> void check(int element, const ElementDofData<DofsPerElement>& data) const {
        if (data.num_elements() != num_elements_)
            throw std::invalid_argument("ElementDofMap requires matching topology and data element counts");
        check_element(element);
    }
    void check_element(int element) const {
        if (element < 0 || element >= num_elements_)
            throw std::out_of_range("ElementDofMap: element is out of range");
    }
    int num_elements_;
};

} // namespace litepgo
