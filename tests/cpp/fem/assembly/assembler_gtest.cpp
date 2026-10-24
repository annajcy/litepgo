#include "litepgo/fem/assembly/assembler.hpp"
#include "litepgo/fem/dof/vertex_dof_map.hpp"
#include "litepgo/linalg/sparse_pattern.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <limits>

namespace {
using namespace litepgo;

TEST(Assembler, SharedHexVerticesMatchDenseScatterAndGlobalFiniteDifferences) {
    const HexTopology topology(12, {{1, 3, 9, 4, 7, 6, 0, 10}, {3, 8, 5, 9, 6, 2, 11, 0}});
    const VertexDofMap map(topology);
    const VertexDofData<3> data(Vec::LinSpaced(36, -0.2, 0.4));
    const Assembler assembler(36, {map.dof_indices(0, data), map.dof_indices(1, data)});
    SpMat hessian = assembler.hessian_pattern();
    EXPECT_TRUE(is_canonical(hessian));
    EXPECT_EQ(hessian.nonZeros(), 2 * 24 * 24 - 12 * 12);
    EXPECT_TRUE(Eigen::MatrixXd(hessian).isZero());
    Vec gradient = Vec::Zero(36), expected_gradient = Vec::Zero(36);
    Eigen::MatrixXd expected_hessian = Eigen::MatrixXd::Zero(36, 36);
    const auto stiffness = [](int e) {
        Eigen::MatrixXd matrix = Eigen::MatrixXd::Constant(24, 24, 0.1 * (e + 1));
        matrix.diagonal().array() += e + 2;
        return matrix;
    };
    for (int e = 0; e < 2; ++e) {
        const Eigen::MatrixXd local_hessian = stiffness(e);
        const Eigen::MatrixXd coefficients = map.gather(e, data);
        const Vec local_gradient = local_hessian * coefficients.reshaped<Eigen::RowMajor>();
        assembler.add_gradient(e, local_gradient, gradient);
        assembler.add_hessian(e, local_hessian, hessian);
        const auto& vertices = topology.cell_vertices(e);
        for (int i = 0; i < 24; ++i) {
            const int row = 3 * vertices[static_cast<std::size_t>(i / 3)] + i % 3;
            expected_gradient[row] += local_gradient[i];
            for (int j = 0; j < 24; ++j) {
                const int col = 3 * vertices[static_cast<std::size_t>(j / 3)] + j % 3;
                expected_hessian(row, col) += local_hessian(i, j);
            }
        }
    }
    EXPECT_TRUE(gradient.isApprox(expected_gradient, 1e-14));
    EXPECT_TRUE(Eigen::MatrixXd(hessian).isApprox(expected_hessian, 1e-14));
    EXPECT_TRUE(same_pattern(hessian, assembler.hessian_pattern()));
    EXPECT_TRUE(Eigen::MatrixXd(assembler.hessian_pattern()).isZero());

    const auto total = [&](const Vec& x) {
        const VertexDofData<3> state(x);
        double energy = 0;
        for (int e = 0; e < 2; ++e) {
            const Eigen::MatrixXd coefficients = map.gather(e, state);
            const Vec local = coefficients.reshaped<Eigen::RowMajor>();
            energy += 0.5 * local.dot(stiffness(e) * local);
        }
        return energy;
    };
    const Vec x = data.values();
    const double step = 1e-4;
    for (int i = 0; i < 36; ++i) {
        Vec plus = x, minus = x;
        plus[i] += step;
        minus[i] -= step;
        EXPECT_NEAR(gradient[i], (total(plus) - total(minus)) / (2 * step), 1e-9);
        for (int j = 0; j < 36; ++j) {
            Vec pp = plus, pm = plus, mp = minus, mm = minus;
            pp[j] += step;
            pm[j] -= step;
            mp[j] += step;
            mm[j] -= step;
            EXPECT_NEAR(hessian.coeff(i, j), (total(pp) - total(pm) - total(mp) + total(mm)) / (4 * step * step), 1e-6);
        }
    }
}

TEST(Assembler, AddsWithoutClearingAndPreservesLocalRowColumnOrder) {
    const Assembler assembler(4, {{2, 0}, {2, 0}, {1, 1}});
    Vec gradient = Vec::Constant(4, 10);
    SpMat hessian = assembler.hessian_pattern();
    std::fill_n(hessian.valuePtr(), hessian.nonZeros(), 10.0);
    Eigen::Matrix2d local;
    local << 1, 2, 3, 4;
    for (int e = 0; e < 3; ++e) {
        assembler.add_gradient(e, Eigen::Vector2d(1, 2), gradient);
        assembler.add_hessian(e, local, hessian);
    }
    EXPECT_EQ(gradient, Eigen::Vector4d(14, 13, 12, 10));
    EXPECT_EQ(hessian.coeff(2, 0), 14);
    EXPECT_EQ(hessian.coeff(0, 2), 16);
    EXPECT_EQ(hessian.coeff(1, 1), 20);
    EXPECT_EQ(hessian.coeff(3, 3), 0);
}

TEST(Assembler, RejectsInvalidInputsBeforeWriting) {
    EXPECT_THROW((Assembler(-1, {})), std::invalid_argument);
    EXPECT_THROW((Assembler(2, {{-1}})), std::invalid_argument);
    EXPECT_THROW((Assembler(2, {{2}})), std::invalid_argument);
    const Assembler assembler(3, {{0, 2}});
    Vec gradient = Vec::Ones(3);
    SpMat hessian = assembler.hessian_pattern();
    const Eigen::Matrix2d local = Eigen::Matrix2d::Identity();
    for (int e : {-1, 1}) {
        EXPECT_THROW(assembler.add_gradient(e, Eigen::Vector2d::Ones(), gradient), std::out_of_range);
        EXPECT_THROW(assembler.add_hessian(e, local, hessian), std::out_of_range);
    }
    EXPECT_THROW(assembler.add_gradient(0, Vec::Zero(1), gradient), std::invalid_argument);
    Vec wrong_size = Vec::Ones(2);
    EXPECT_THROW(assembler.add_gradient(0, Eigen::Vector2d::Ones(), wrong_size), std::invalid_argument);
    EXPECT_THROW(assembler.add_hessian(0, Eigen::MatrixXd::Zero(2, 1), hessian), std::invalid_argument);
    Eigen::Vector2d bad(1, std::numeric_limits<double>::quiet_NaN());
    EXPECT_THROW(assembler.add_gradient(0, bad, gradient), std::invalid_argument);
    Eigen::Matrix2d bad_matrix = local;
    bad_matrix(1, 1) = std::numeric_limits<double>::infinity();
    EXPECT_THROW(assembler.add_hessian(0, bad_matrix, hessian), std::invalid_argument);
    EXPECT_EQ(gradient, Vec::Ones(3));
    EXPECT_TRUE(Eigen::MatrixXd(hessian).isZero());
    SpMat wrong(3, 3);
    std::vector<Triplet> entries{{0, 0, 7}, {1, 0, 7}, {0, 2, 7}, {2, 2, 7}};
    wrong.setFromTriplets(entries.begin(), entries.end());
    const Eigen::MatrixXd before(wrong);
    EXPECT_THROW(assembler.add_hessian(0, local, wrong), std::invalid_argument);
    EXPECT_EQ(Eigen::MatrixXd(wrong), before);
    hessian.uncompress();
    EXPECT_THROW(assembler.add_hessian(0, local, hessian), std::invalid_argument);
}

TEST(Assembler, AllowsEmptyPlansAndEmptyElements) {
    const Assembler empty(0, {});
    EXPECT_EQ(empty.hessian_pattern().rows(), 0);
    EXPECT_TRUE(is_canonical(empty.hessian_pattern()));
    Vec gradient(0);
    EXPECT_THROW(empty.add_gradient(0, Vec(0), gradient), std::out_of_range);
    const Assembler assembler(3, {{}});
    SpMat hessian = assembler.hessian_pattern();
    Vec values = Vec::Ones(3);
    assembler.add_gradient(0, Vec(0), values);
    assembler.add_hessian(0, Eigen::MatrixXd(0, 0), hessian);
    EXPECT_EQ(values, Vec::Ones(3));
    EXPECT_EQ(hessian.nonZeros(), 0);
}
} // namespace
