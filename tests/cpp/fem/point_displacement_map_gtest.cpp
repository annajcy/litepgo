#include "litepgo/fem/point_displacement_map.hpp"
#include "litepgo/fem/response/interpolated_response.hpp"
#include <gtest/gtest.h>
#include <memory>
using namespace litepgo;
using namespace litepgo::fem;
TEST(PointDisplacementMap, ExtractedTetWeightsAndValidation) {
    Eigen::MatrixXd X(4, 3);
    X << 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4;
    std::vector<std::shared_ptr<litepgo::energy::Response>> fields{
        std::make_shared<InterpolatedResponse<TetP1Basis>>(X)};
    auto embedding = ElementLocator<TetP1Basis>(fields).locate(X);
    dof::VertexDofMap map(mesh::TetTopology(4, {{0, 1, 2, 3}}), 3);
    const auto W = build_point_displacement_map<TetP1Basis>(embedding, map, fields);
    EXPECT_TRUE(Eigen::MatrixXd(W).isIdentity());
    embedding.element_ids[0] = -1;
    EXPECT_THROW(build_point_displacement_map<TetP1Basis>(embedding, map, fields), std::out_of_range);
    embedding.element_ids[0] = 0;
    embedding.xi(0, 0) = 2;
    EXPECT_THROW(build_point_displacement_map<TetP1Basis>(embedding, map, fields), std::invalid_argument);
    embedding.xi(0, 0) = NAN;
    EXPECT_THROW(build_point_displacement_map<TetP1Basis>(embedding, map, fields), std::invalid_argument);
    embedding.xi.resize(0, 3);
    EXPECT_THROW(build_point_displacement_map<TetP1Basis>(embedding, map, fields), std::invalid_argument);
}
