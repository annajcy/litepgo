#include "litepgo/fem/element_locator.hpp"
#include "litepgo/fem/response/interpolated_response.hpp"
#include <gtest/gtest.h>
#include <memory>
using namespace litepgo::fem;
TEST(ElementLocator, TetBoundaryInteriorAndOutside) {
    Eigen::MatrixXd X(4, 3);
    X << 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4;
    auto field = std::make_shared<InterpolatedResponse<TetP1Basis>>(X);
    ElementLocator<TetP1Basis> locator({field, field});
    Eigen::MatrixXd points(5, 3);
    points.topRows(4) = X;
    points.row(4) << 0.4, 0.6, 0.8;
    auto result = locator.locate(points);
    EXPECT_EQ(result.element_ids, Eigen::VectorXi::Zero(5));
    for (int i = 0; i < 5; ++i)
        EXPECT_TRUE(field->eval_value(result.xi.row(i).transpose()).isApprox(points.row(i).transpose(), 1e-9));
    points.row(0) << 1.9, 2.9, 3.9;
    EXPECT_THROW(locator.locate(points), std::runtime_error);
    points.row(0) << -10, 0, 0;
    EXPECT_THROW(locator.locate(points), std::runtime_error);
}
TEST(ElementLocator, WarpedHexInverseAndDegenerateRejection) {
    Eigen::MatrixXd X(8, 3);
    for (int i = 0; i < 8; ++i)
        X.row(i) = HexQ1Basis::coefficient_info(i).point.transpose();
    X(6, 0) += 0.3;
    X(6, 2) += 0.4;
    auto field = std::make_shared<InterpolatedResponse<HexQ1Basis>>(X);
    ElementLocator<HexQ1Basis> locator({field});
    Eigen::MatrixXd xi(3, 3);
    xi << 0.2, 0.3, 0.4, 0, 0.8, 1, 1, 1, 1;
    Eigen::MatrixXd points(3, 3);
    for (int i = 0; i < 3; ++i)
        points.row(i) = field->eval_value(xi.row(i).transpose()).transpose();
    EXPECT_TRUE(locator.locate(points).xi.isApprox(xi, 1e-8));
    X.col(2).setZero();
    ElementLocator<HexQ1Basis> flat({std::make_shared<InterpolatedResponse<HexQ1Basis>>(X)});
    points.setZero();
    EXPECT_THROW(flat.locate(points), std::runtime_error);
}
