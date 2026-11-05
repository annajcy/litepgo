#include "litepgo/fem/point_displacement_map.hpp"
#include "litepgo/fem/sample_boundary.hpp"
#include "litepgo/mesh/extract_surface.hpp"
#include <gtest/gtest.h>
#include <map>
using namespace litepgo;
using namespace litepgo::fem;
TEST(SampleBoundary, MatchesExtractionAndWeldsEdges) {
    Eigen::MatrixXd X(8, 3);
    for (int i = 0; i < 8; ++i)
        X.row(i) = HexQ1Basis::coefficient_info(i).point.transpose();
    mesh::HexTopology topology(8, {{0, 1, 2, 3, 4, 5, 6, 7}});
    std::vector<InterpolatedField3D<HexQ1Basis>> fields{InterpolatedField3D<HexQ1Basis>(X)};
    const auto extracted = mesh::extract_surface(mesh::HexMesh{X.reshaped<Eigen::RowMajor>(), topology});
    const auto one = sample_boundary<HexQ1Basis>(topology, fields, 1);
    EXPECT_EQ(one.mesh.positions, extracted.positions);
    ASSERT_EQ(one.mesh.topology.num_cells(), extracted.topology.num_cells());
    for (int i = 0; i < extracted.topology.num_cells(); ++i)
        EXPECT_EQ(one.mesh.topology.cell_vertices(i), extracted.topology.cell_vertices(i));
    for (int n : {2, 3, 4}) {
        const auto sampled = sample_boundary<HexQ1Basis>(topology, fields, n);
        EXPECT_EQ(sampled.mesh.topology.num_vertices(), 6 * n * n + 2);
        EXPECT_EQ(sampled.mesh.topology.num_cells(), 12 * n * n);
        std::map<std::pair<int, int>, int> edges;
        for (int i = 0; i < sampled.mesh.topology.num_cells(); ++i) {
            const auto& t = sampled.mesh.topology.cell_vertices(i);
            for (int j = 0; j < 3; ++j)
                ++edges[{t[j], t[(j + 1) % 3]}];
        }
        for (const auto& [edge, count] : edges) {
            EXPECT_EQ(count, 1);
            EXPECT_EQ(edges.at({edge.second, edge.first}), 1);
        }
    }
    const auto W =
        build_point_displacement_map<HexHermiteBasis>(one.embedding, dof::VertexDofMap(topology, 24), fields);
    EXPECT_EQ(W.nonZeros(), 24);
    EXPECT_THROW(sample_boundary<HexQ1Basis>(topology, fields, 0), std::invalid_argument);
    EXPECT_THROW(sample_boundary<HexQ1Basis>(topology, fields, 100000), std::length_error);
    EXPECT_THROW(sample_boundary<HexQ1Basis>(topology, {}, 2), std::invalid_argument);
}
