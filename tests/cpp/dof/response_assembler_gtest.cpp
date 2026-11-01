#include "litepgo/dof/response_assembler.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <tbb/global_control.h>

using namespace litepgo;
using dof::ResponseAssembler;

TEST(ResponseAssembler, RectangularSharedRowsMatchSerialAcrossThreadCounts) {
    ResponseAssembler::Indices rows, cols;
    std::vector<Vec> values;
    std::vector<Eigen::MatrixXd> jacobians;
    Vec expected = Vec::Zero(1003);
    Eigen::MatrixXd expected_J = Eigen::MatrixXd::Zero(1003, 2);
    for (int e = 0; e < 1000; ++e) {
        rows.push_back({e + 2, e});
        cols.push_back(e % 2 ? std::vector<int>{1, 0} : std::vector<int>{0, 1});
        values.push_back((Vec(2) << e * .125, -e * .25).finished());
        Eigen::MatrixXd J(2, 2);
        J << e + 1, -.5 * e, 2 * e, 3;
        jacobians.push_back(J);
        for (int r = 0; r < 2; ++r) {
            expected[rows.back()[r]] += values.back()[r];
            for (int c = 0; c < 2; ++c)
                expected_J(rows.back()[r], cols.back()[c]) += J(r, c);
        }
    }
    ResponseAssembler assembler(1003, 2, rows, cols);
    std::vector<ResponseAssembler::ElementView> views;
    for (int e = 0; e < 1000; ++e)
        views.emplace_back(values[e], jacobians[e]);
    for (int threads : {1, 2, 4}) {
        tbb::global_control limit(tbb::global_control::max_allowed_parallelism, threads);
        SpMat J = assembler.jacobian_pattern();
        Vec value = Vec::Constant(1003, 123);
        for (int repeat = 0; repeat < 3; ++repeat) {
            assembler.assemble(views, value, &J);
            EXPECT_EQ((value - expected).norm(), 0);
            EXPECT_EQ((Eigen::MatrixXd(J) - expected_J).norm(), 0);
        }
        assembler.assemble(views, value);
        EXPECT_EQ((value - expected).norm(), 0);
    }
}

TEST(ResponseAssembler, StridedBatchViewsAndOriginalElementSummationOrder) {
    ResponseAssembler assembler(1, 2, {{0}, {0}, {0}}, {{0, 1}, {0, 1}, {0, 1}});
    // Strided values from a column-major matrix row, Jacobians from row-major padded blocks.
    Eigen::MatrixXd storage(2, 3);
    storage.row(0) << 1e16, -1e16, 1;
    storage.row(1).setConstant(99);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrices(3, 4);
    matrices << 99, 1e16, 2, 99, 99, -1e16, 3, 99, 99, 1, 4, 99;
    std::vector<ResponseAssembler::ElementView> views;
    for (int e = 0; e < 3; ++e)
        views.emplace_back(storage.row(0).segment(e, 1), matrices.block(e, 1, 1, 2));
    SpMat J = assembler.jacobian_pattern();
    Vec value;
    assembler.assemble(views, value, &J);
    EXPECT_EQ(value[0], 1);
    EXPECT_EQ(J.coeff(0, 0), 1);
    EXPECT_EQ(J.coeff(0, 1), 9);
}

TEST(ResponseAssembler, ValidationBeforeWritesEmptyAndOverflow) {
    EXPECT_THROW((ResponseAssembler(1, 1, {{0}}, {})), std::invalid_argument);
    EXPECT_THROW((ResponseAssembler(-1, 1, {}, {})), std::invalid_argument);
    EXPECT_THROW((ResponseAssembler(1, 1, {{1}}, {{0}})), std::invalid_argument);
    EXPECT_THROW((ResponseAssembler(1, 1, {{0}}, {{-1}})), std::invalid_argument);
    EXPECT_NO_THROW((ResponseAssembler(1, 1, {{0, 0}}, {{0}})));
    ResponseAssembler assembler(2, 1, {{0}, {0}}, {{0}, {0}});
    EXPECT_THROW(assembler.input_indices(-1), std::out_of_range);
    EXPECT_THROW(assembler.output_indices(2), std::out_of_range);
    Vec local = Vec::Ones(1), wrong = Vec::Ones(2), global = Vec::Constant(2, 7);
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Ones(1, 1);
    SpMat J = assembler.jacobian_pattern();
    J.coeffs().setConstant(9);
    std::vector<ResponseAssembler::ElementView> views;
    views.emplace_back(local, matrix);
    EXPECT_THROW(assembler.assemble(views, global, &J), std::invalid_argument);
    views.emplace_back(wrong, matrix);
    EXPECT_THROW(assembler.assemble(views, global, &J), std::invalid_argument);
    EXPECT_TRUE((global.array() == 7).all());
    EXPECT_TRUE((J.coeffs().array() == 9).all());
    views.clear();
    views.emplace_back(local, matrix);
    views.emplace_back(local, matrix);
    matrix(0, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(assembler.assemble(views, global, &J), std::invalid_argument);
    EXPECT_TRUE((global.array() == 7).all());
    EXPECT_TRUE((J.coeffs().array() == 9).all());
    assembler.assemble(views, global); // Unrequested Jacobians are ignored.
    EXPECT_EQ(global[0], 2);
    matrix(0, 0) = 1;
    SpMat wrong_pattern(2, 1);
    EXPECT_THROW(assembler.assemble(views, global, &wrong_pattern), std::invalid_argument);
    local[0] = std::numeric_limits<double>::max();
    EXPECT_THROW(assembler.assemble(views, global), std::domain_error);
    local[0] = 0;
    matrix(0, 0) = std::numeric_limits<double>::max();
    EXPECT_THROW(assembler.assemble(views, global, &J), std::domain_error);
    ResponseAssembler empty(3, 2, {}, {});
    J = empty.jacobian_pattern();
    empty.assemble({}, global, &J);
    EXPECT_EQ(global.size(), 3);
    EXPECT_EQ(global.norm(), 0);
    EXPECT_EQ(J.nonZeros(), 0);
}

TEST(ResponseAssembler, PaddedRowMajorJacobianAndStridedVectorViews) {
    ResponseAssembler assembler(2, 3, {{1, 0}}, {{2, 0, 1}});
    Eigen::MatrixXd values(2, 2);
    values << 5, 7, 99, 99;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrices(4, 5);
    matrices.setConstant(99);
    matrices.block(1, 1, 2, 3) << 1, 2, 3, 4, 5, 6;
    std::vector<ResponseAssembler::ElementView> views;
    views.emplace_back(values.row(0), matrices.block(1, 1, 2, 3));
    SpMat J = assembler.jacobian_pattern();
    Vec value;
    assembler.assemble(views, value, &J);
    EXPECT_EQ(value[0], 7);
    EXPECT_EQ(value[1], 5);
    Eigen::Matrix<double, 2, 3> expected;
    expected << 5, 6, 4, 2, 3, 1;
    EXPECT_EQ((Eigen::MatrixXd(J) - expected).norm(), 0);
}

TEST(ResponseAssembler, RepeatedRowsAndColumnsAccumulateRectangularBlocksDeterministically) {
    const ResponseAssembler::Indices rows{{0, 0, 1}, {0, 1, 1}};
    const ResponseAssembler::Indices cols{{1, 1}, {0, 1}};
    ResponseAssembler assembler(2, 2, rows, cols);
    std::vector<Vec> values{(Vec(3) << 1, 2, 3).finished(), (Vec(3) << 4, 5, 6).finished()};
    std::vector<Eigen::MatrixXd> matrices(2, Eigen::MatrixXd(3, 2));
    matrices[0] << 1, 2, 3, 4, 5, 6;
    matrices[1] << 7, 8, 9, 10, 11, 12;
    Eigen::Matrix2d expected = Eigen::Matrix2d::Zero();
    Vec expected_value = Vec::Zero(2);
    for (int e = 0; e < 2; ++e)
        for (int r = 0; r < 3; ++r) {
            expected_value[rows[e][r]] += values[e][r];
            for (int c = 0; c < 2; ++c)
                expected(rows[e][r], cols[e][c]) += matrices[e](r, c);
        }
    std::vector<ResponseAssembler::ElementView> views(2);
    for (int e = 0; e < 2; ++e)
        views[e] = ResponseAssembler::ElementView(values[e], matrices[e]);
    for (int threads : {1, 2, 4}) {
        tbb::global_control limit(tbb::global_control::max_allowed_parallelism, threads);
        SpMat J = assembler.jacobian_pattern();
        EXPECT_TRUE(J.coeffs().isZero());
        Vec v;
        assembler.assemble(views, v, &J);
        EXPECT_TRUE(v.isApprox(expected_value));
        EXPECT_TRUE(Eigen::MatrixXd(J).isApprox(expected));
    }
}
