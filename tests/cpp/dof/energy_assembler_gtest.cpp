#include "litepgo/dof/energy_assembler.hpp"
#include "litepgo/dof/vertex_dof_map.hpp"
#include "litepgo/linalg/sparse_pattern.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace litepgo::dof::test {

namespace {

TEST(EnergyAssembler, SharedHexVerticesMatchDenseScatterAndGlobalFiniteDifferences) {
    const mesh::HexTopology topology(12, {{1, 3, 9, 4, 7, 6, 0, 10}, {3, 8, 5, 9, 6, 2, 11, 0}});
    const VertexDofMap map(topology, 3);
    const Vec data(Vec::LinSpaced(36, -0.2, 0.4));
    const EnergyAssembler assembler(36, {map.dof_indices(0), map.dof_indices(1)});
    SpMat hessian = assembler.hessian_pattern();
    EXPECT_TRUE(linalg::is_canonical(hessian));
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
    EXPECT_TRUE(linalg::same_pattern(hessian, assembler.hessian_pattern()));
    EXPECT_TRUE(Eigen::MatrixXd(assembler.hessian_pattern()).isZero());

    const auto total = [&](const Vec& x) {
        double energy = 0;
        for (int e = 0; e < 2; ++e) {
            const Eigen::MatrixXd coefficients = map.gather(e, x);
            const Vec local = coefficients.reshaped<Eigen::RowMajor>();
            energy += 0.5 * local.dot(stiffness(e) * local);
        }
        return energy;
    };
    const Vec x = data;
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

TEST(EnergyAssembler, AddsWithoutClearingAndPreservesLocalRowColumnOrder) {
    const EnergyAssembler assembler(4, {{2, 0}, {2, 0}, {1, 1}});
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

TEST(EnergyAssembler, RejectsInvalidInputsBeforeWriting) {
    EXPECT_THROW((EnergyAssembler(-1, {})), std::invalid_argument);
    EXPECT_THROW((EnergyAssembler(2, {{-1}})), std::invalid_argument);
    EXPECT_THROW((EnergyAssembler(2, {{2}})), std::invalid_argument);
    const EnergyAssembler assembler(3, {{0, 2}});
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

TEST(EnergyAssembler, AllowsEmptyPlansAndEmptyElements) {
    const EnergyAssembler empty(0, {});
    EXPECT_EQ(empty.hessian_pattern().rows(), 0);
    EXPECT_TRUE(linalg::is_canonical(empty.hessian_pattern()));
    Vec gradient(0);
    EXPECT_THROW(empty.add_gradient(0, Vec(0), gradient), std::out_of_range);
    const EnergyAssembler assembler(3, {{}});
    SpMat hessian = assembler.hessian_pattern();
    Vec values = Vec::Ones(3);
    assembler.add_gradient(0, Vec(0), values);
    assembler.add_hessian(0, Eigen::MatrixXd(0, 0), hessian);
    EXPECT_EQ(values, Vec::Ones(3));
    EXPECT_EQ(hessian.nonZeros(), 0);
}
} // namespace

TEST(EnergyAssembler, AssembleOverwritesWithTheSumOfAddedResults) {
    // Repeated and shared indices, including an element with no DOFs.
    const EnergyAssembler assembler(4, {{2, 0}, {0, 3, 1}, {}, {1, 1}});
    std::vector<Vec> gradients;
    std::vector<Eigen::MatrixXd> hessians;
    Vec expected_gradient = Vec::Zero(4);
    SpMat expected_hessian = assembler.hessian_pattern();
    for (int e = 0; e < assembler.num_elements(); ++e) {
        const auto n = static_cast<Eigen::Index>(assembler.dof_indices(e).size());
        gradients.push_back(Vec::LinSpaced(n, e + 1.0, e + 2.0));
        hessians.push_back(Eigen::MatrixXd::Random(n, n));
        assembler.add_gradient(e, gradients.back(), expected_gradient);
        assembler.add_hessian(e, hessians.back(), expected_hessian);
    }
    Vec gradient = Vec::Constant(7, 9.0); // resized and overwritten
    SpMat hessian = assembler.hessian_pattern();
    hessian.coeffs().setConstant(9.0);
    assembler.assemble(gradients, hessians, &gradient, &hessian);
    EXPECT_TRUE(gradient.isApprox(expected_gradient, 1e-15));
    EXPECT_TRUE(Eigen::MatrixXd(hessian).isApprox(Eigen::MatrixXd(expected_hessian), 1e-15));
    // Either output alone, and neither.
    Vec gradient_only;
    assembler.assemble(gradients, {}, &gradient_only, nullptr);
    EXPECT_EQ(gradient_only, gradient);
    SpMat hessian_only = assembler.hessian_pattern();
    assembler.assemble({}, hessians, nullptr, &hessian_only);
    EXPECT_EQ(Eigen::MatrixXd(hessian_only), Eigen::MatrixXd(hessian));
    assembler.assemble({}, {}, nullptr, nullptr);
}

TEST(EnergyAssembler, AssembleRejectsInvalidInputsBeforeWriting) {
    const EnergyAssembler assembler(3, {{0, 2}, {1}});
    const std::vector<Vec> gradients = {Vec::Ones(2), Vec::Ones(1)};
    const std::vector<Eigen::MatrixXd> hessians = {Eigen::MatrixXd::Ones(2, 2), Eigen::MatrixXd::Ones(1, 1)};
    Vec gradient = Vec::Constant(3, 5.0);
    SpMat hessian = assembler.hessian_pattern();
    hessian.coeffs().setConstant(5.0);
    const auto expect_untouched = [&] {
        EXPECT_EQ(gradient, Vec::Constant(3, 5.0));
        EXPECT_TRUE((hessian.coeffs() == 5.0).all());
    };
    EXPECT_THROW(assembler.assemble(std::span(gradients).first(1), hessians, &gradient, &hessian),
                 std::invalid_argument);
    expect_untouched();
    EXPECT_THROW(assembler.assemble(gradients, std::span(hessians).first(1), &gradient, &hessian),
                 std::invalid_argument);
    expect_untouched();
    const std::vector<Vec> wrong_gradient = {Vec::Ones(2), Vec::Ones(2)};
    EXPECT_THROW(assembler.assemble(wrong_gradient, hessians, &gradient, &hessian), std::invalid_argument);
    expect_untouched();
    const std::vector<Eigen::MatrixXd> wrong_hessian = {Eigen::MatrixXd::Ones(2, 1), Eigen::MatrixXd::Ones(1, 1)};
    EXPECT_THROW(assembler.assemble(gradients, wrong_hessian, &gradient, &hessian), std::invalid_argument);
    expect_untouched();
    SpMat other_pattern(3, 3);
    other_pattern.insert(0, 0) = 5.0;
    EXPECT_THROW(assembler.assemble(gradients, hessians, &gradient, &other_pattern), std::invalid_argument);
    expect_untouched();
}

TEST(EnergyAssembler, AssembleRejectsNonFiniteSums) {
    const EnergyAssembler assembler(2, {{0, 1}});
    std::vector<Vec> gradients = {Vec::Ones(2)};
    std::vector<Eigen::MatrixXd> hessians = {Eigen::MatrixXd::Ones(2, 2)};
    gradients[0][1] = std::numeric_limits<double>::quiet_NaN();
    hessians[0](1, 0) = std::numeric_limits<double>::infinity();
    Vec gradient;
    SpMat hessian = assembler.hessian_pattern();
    EXPECT_THROW(assembler.assemble(gradients, {}, &gradient, nullptr), std::invalid_argument);
    EXPECT_THROW(assembler.assemble({}, hessians, nullptr, &hessian), std::invalid_argument);
}

} // namespace litepgo::dof::test

TEST(EnergyAssemblerViews, StridedBlocksPreserveOrientationAndSharedAccumulation) {
    using namespace litepgo;
    const dof::EnergyAssembler assembler(3, {{0, 1}, {1, 2}});
    Eigen::MatrixXd gradients = Eigen::MatrixXd::Constant(2, 4, -99);
    gradients.row(0).head(2) << 1, 2;
    gradients.row(1).head(2) << 3, 4;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> blocks =
        Eigen::MatrixXd::Constant(6, 5, -99);
    blocks.block(0, 1, 2, 2) << 1, 2, 3, 4;
    blocks.block(3, 1, 2, 2) << 5, 6, 7, 8;
    std::vector<dof::EnergyAssembler::ElementView> views;
    views.emplace_back(gradients.row(0).head(2), blocks.block(0, 1, 2, 2));
    views.emplace_back(gradients.row(1).head(2), blocks.block(3, 1, 2, 2));
    Vec g;
    SpMat H = assembler.hessian_pattern();
    assembler.assemble(views, &g, &H);
    EXPECT_TRUE(g.isApprox((Vec(3) << 1, 5, 4).finished()));
    Eigen::Matrix3d expected;
    expected << 1, 2, 0, 3, 9, 6, 0, 7, 8;
    EXPECT_TRUE(Eigen::MatrixXd(H).isApprox(expected));
    // Views must observe current storage, rather than own an old copy.
    blocks(0, 1) = 11;
    assembler.assemble(views, nullptr, &H);
    EXPECT_DOUBLE_EQ(H.coeff(0, 0), 11);
    const Vec saved = g;
    views.clear();
    views.emplace_back(gradients.row(0).head(1), blocks.block(0, 1, 2, 2));
    views.emplace_back(gradients.row(1).head(2), blocks.block(3, 1, 2, 2));
    EXPECT_THROW(assembler.assemble(views, &g, &H), std::invalid_argument);
    EXPECT_TRUE(g.isApprox(saved));
}

TEST(EnergyAssemblerViews, InvalidInputsLeaveBothOutputsUntouchedAndRetrySucceeds) {
    using namespace litepgo;
    dof::EnergyAssembler assembler(2, {{0, 1}});
    Vec local = Vec::Ones(2), g = Vec::Constant(2, 7);
    Eigen::MatrixXd Hlocal = Eigen::MatrixXd::Identity(2, 2);
    SpMat H = assembler.hessian_pattern();
    H.coeffs().setConstant(9);
    std::vector<dof::EnergyAssembler::ElementView> views;
    views.emplace_back(local, Hlocal);
    for (bool bad_gradient : {true, false}) {
        local.setOnes();
        Hlocal.setIdentity();
        if (bad_gradient)
            local[1] = std::numeric_limits<double>::quiet_NaN();
        else
            Hlocal(1, 0) = std::numeric_limits<double>::infinity();
        EXPECT_THROW(assembler.assemble(views, &g, &H), std::invalid_argument);
        EXPECT_TRUE(g.isConstant(7));
        EXPECT_TRUE(H.coeffs().isConstant(9));
    }
    // The unrequested invalid Hessian must not be read.
    assembler.assemble(views, &g, nullptr);
    EXPECT_TRUE(g.isOnes());
    Hlocal.setIdentity();
    assembler.assemble(views, &g, &H);
    EXPECT_TRUE(Eigen::MatrixXd(H).isIdentity());
}

TEST(EnergyAssemblerViews, PackedAndPaddedMatricesValidateWithoutReadingPadding) {
    using namespace litepgo;
    const dof::EnergyAssembler assembler(3, {{0, 1, 2}});
    const Vec local_g = Vec::Ones(3);
    for (const auto& strides : {std::pair{3, 1}, std::pair{1, 3}, std::pair{5, 1}, std::pair{1, 5}, std::pair{2, 7}}) {
        const auto [row_stride, col_stride] = strides;
        Vec storage = Vec::Constant(2 * (row_stride + col_stride) + 3, std::numeric_limits<double>::quiet_NaN());
        Eigen::Map<Eigen::MatrixXd, 0, Eigen::Stride<-1, -1>> local_H(storage.data() + 1, 3, 3,
                                                                      Eigen::Stride<-1, -1>(col_stride, row_stride));
        local_H.setIdentity();
        const std::vector<dof::EnergyAssembler::ElementView> views{{local_g, local_H}};
        Vec g;
        SpMat H = assembler.hessian_pattern();
        assembler.assemble(views, &g, &H);
        EXPECT_TRUE(g.isOnes());
        EXPECT_TRUE(Eigen::MatrixXd(H).isIdentity());
        g.setConstant(7);
        H.coeffs().setConstant(9);
        local_H(2, 1) = std::numeric_limits<double>::infinity();
        EXPECT_THROW(assembler.assemble(views, &g, &H), std::invalid_argument);
        EXPECT_TRUE(g.isConstant(7));
        EXPECT_TRUE(H.coeffs().isConstant(9));
        local_H.setIdentity();
        EXPECT_NO_THROW(assembler.assemble(views, &g, &H));
    }
}
