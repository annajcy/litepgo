#pragma once
#include "litepgo/mesh/mesh_topology.hpp"
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <vector>
namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::dof::binding {
inline Eigen::VectorXi to_array(const std::vector<int>& indices) {
    return Eigen::Map<const Eigen::VectorXi>(indices.data(), static_cast<Eigen::Index>(indices.size()));
}

// Members shared by every DOF map.
template <class Map> nb::class_<Map> bind_map(nb::module_& m, const char* name) {
    return nb::class_<Map>(m, name)
        .def_prop_ro("num_elements", &Map::num_elements)
        .def_prop_ro("num_dofs", &Map::num_dofs)
        .def(
            "dof_indices", [](const Map& map, int element) { return to_array(map.dof_indices(element)); }, "element"_a,
            "Global DOFs of one element in local order (a copy).")
        .def("gather", &Map::gather, "element"_a, "x"_a, "The element's entries of a global vector.");
}

// The C++ constructors take any MeshTopology<K>.
template <class Cls> void def_topology_init(Cls& cls, const char* width) {
    cls.def(nb::init<const mesh::MeshTopology<3>&, int>(), "topology"_a, nb::arg(width))
        .def(nb::init<const mesh::MeshTopology<4>&, int>(), "topology"_a, nb::arg(width))
        .def(nb::init<const mesh::MeshTopology<8>&, int>(), "topology"_a, nb::arg(width));
}
} // namespace litepgo::dof::binding
