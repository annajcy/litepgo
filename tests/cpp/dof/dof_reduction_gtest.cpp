#include "litepgo/dof/dof_reduction.hpp"

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

namespace litepgo::dof::test {

namespace {

std::vector<int> as_vector(std::span<const int> values) {
    return {values.begin(), values.end()};
}

// A dense symmetric matrix with distinct entries, as a sparse one.
SpMat numbered_matrix(int n) {
    Eigen::MatrixXd dense(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            dense(i, j) = 1.0 + i + j + 0.1 * i * j;
        }
    }
    SpMat A = dense.sparseView();
    A.makeCompressed();
    return A;
}

} // namespace

TEST(DofReduction, SplitsFixedAndFreeDofs) {
    const DofReduction reduction(6, {4, 1});
    EXPECT_EQ(reduction.num_dofs(), 6);
    EXPECT_EQ(reduction.num_free(), 4);
    EXPECT_EQ(as_vector(reduction.fixed_dofs()), (std::vector<int>{1, 4}));
    EXPECT_EQ(as_vector(reduction.free_dofs()), (std::vector<int>{0, 2, 3, 5}));
}

TEST(DofReduction, FreeIndexInvertsFreeDofs) {
    const DofReduction reduction(6, {4, 1});
    for (int i = 0; i < reduction.num_free(); ++i) {
        EXPECT_EQ(reduction.free_index(reduction.free_dofs()[static_cast<size_t>(i)]), i);
    }
    EXPECT_EQ(reduction.free_index(1), -1);
    EXPECT_EQ(reduction.free_index(4), -1);
    EXPECT_THROW(reduction.free_index(6), std::out_of_range);
    EXPECT_THROW(reduction.free_index(-1), std::out_of_range);
}

TEST(DofReduction, HandlesNoAndAllFixedDofs) {
    const DofReduction none(3, {});
    EXPECT_EQ(as_vector(none.free_dofs()), (std::vector<int>{0, 1, 2}));
    const DofReduction all(3, {2, 0, 1});
    EXPECT_EQ(all.num_free(), 0);
    EXPECT_EQ(all.reduce(Vec::Ones(3)).size(), 0);
    EXPECT_EQ(all.reduce(numbered_matrix(3)).rows(), 0);
    const DofReduction empty(0, {});
    EXPECT_EQ(empty.num_free(), 0);
}

TEST(DofReduction, RejectsInvalidFixedDofs) {
    EXPECT_THROW(DofReduction(3, {1, 1}), std::invalid_argument);
    EXPECT_THROW(DofReduction(3, {3}), std::invalid_argument);
    EXPECT_THROW(DofReduction(3, {-1}), std::invalid_argument);
    EXPECT_THROW(DofReduction(-1, {}), std::invalid_argument);
}

TEST(DofReduction, ReduceAndExpandRoundTrip) {
    const DofReduction reduction(6, {4, 1});
    const Vec full = Vec::LinSpaced(6, 10.0, 15.0);
    const Vec reduced = reduction.reduce(full);
    EXPECT_EQ(reduced, (Vec(4) << 10.0, 12.0, 13.0, 15.0).finished());

    Vec expanded = Vec::Constant(6, -7.0);
    reduction.expand(reduced, expanded);
    EXPECT_EQ(expanded, (Vec(6) << 10.0, -7.0, 12.0, 13.0, -7.0, 15.0).finished());
}

TEST(DofReduction, ChecksVectorArguments) {
    const DofReduction reduction(6, {4, 1});
    Vec reduced(4);
    EXPECT_THROW(reduction.reduce(Vec::Ones(5), reduced), std::invalid_argument);
    Vec short_reduced(3);
    EXPECT_THROW(reduction.reduce(Vec::Ones(6), short_reduced), std::invalid_argument);
    Vec full = Vec::Ones(6);
    EXPECT_THROW(reduction.expand(Vec::Ones(3), full), std::invalid_argument);
    EXPECT_THROW(reduction.expand(full.head(4), full), std::invalid_argument);
    EXPECT_THROW(reduction.reduce(full, full.segment(1, 4)), std::invalid_argument);
}

TEST(DofReduction, ReducesThePrincipalSubmatrix) {
    const DofReduction reduction(6, {4, 1});
    const SpMat A = numbered_matrix(6);
    const SpMat reduced = reduction.reduce(A);
    ASSERT_EQ(reduced.rows(), 4);
    ASSERT_EQ(reduced.cols(), 4);
    const Eigen::MatrixXd dense = Eigen::MatrixXd(A);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_EQ(reduced.coeff(i, j), dense(reduction.free_dofs()[static_cast<size_t>(i)],
                                                 reduction.free_dofs()[static_cast<size_t>(j)]));
        }
    }
}

TEST(DofReduction, ReduceChecksTheMatrix) {
    const DofReduction reduction(4, {0});
    EXPECT_THROW(reduction.reduce(numbered_matrix(5)), std::invalid_argument);
    SpMat uncompressed(4, 4);
    uncompressed.reserve(Eigen::VectorXi::Constant(4, 1));
    uncompressed.insert(1, 1) = 1.0;
    ASSERT_FALSE(uncompressed.isCompressed());
    EXPECT_THROW(reduction.reduce(uncompressed), std::invalid_argument);
}

} // namespace litepgo::dof::test
