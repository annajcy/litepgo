#pragma once
#include "litepgo/mesh/mesh_topology.hpp"
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::mesh::binding {
template <int K> using Cells = Eigen::Matrix<int, Eigen::Dynamic, K, Eigen::RowMajor>;
template <int K> MeshTopology<K> make_topology(int num_vertices, const Cells<K>& cells) {
    std::vector<typename MeshTopology<K>::Cell> connectivity(static_cast<std::size_t>(cells.rows()));
    for (Eigen::Index e = 0; e < cells.rows(); ++e)
        for (int j = 0; j < K; ++j)
            connectivity[static_cast<std::size_t>(e)][static_cast<std::size_t>(j)] = cells(e, j);
    return MeshTopology<K>(num_vertices, std::move(connectivity));
}

template <int K> Cells<K> cells_of(const MeshTopology<K>& topology) {
    Cells<K> cells(topology.num_cells(), K);
    for (int e = 0; e < topology.num_cells(); ++e)
        for (int j = 0; j < K; ++j)
            cells(e, j) = topology.cell_vertices(e)[static_cast<std::size_t>(j)];
    return cells;
}
} // namespace litepgo::mesh::binding
