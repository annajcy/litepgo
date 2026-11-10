#include "litepgo/dof/reduced_matrix.hpp"

#include "litepgo/dof/dof_reduction.hpp"
#include "litepgo/linalg/sparse_pattern.hpp"

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

namespace litepgo::dof::test {

namespace {

// Tridiagonal symmetric matrix whose values depend on scale.
SpMat tridiagonal(int n, double scale) {
    std::vector<Triplet> entries;
    for (int i = 0; i < n; ++i) {
        entries.emplace_back(i, i, scale * (2.0 + i));
        if (i + 1 < n) {
            entries.emplace_back(i, i + 1, -scale);
            entries.emplace_back(i + 1, i, -scale);
        }
    }
    SpMat A(n, n);
    A.setFromTriplets(entries.begin(), entries.end());
    return A;
}

void expect_same(const SpMat& actual, const SpMat& expected) {
    EXPECT_EQ(Eigen::MatrixXd(actual), Eigen::MatrixXd(expected));
}

} // namespace

TEST(ReducedMatrix, MatchesReduce) {
    const DofReduction reduction(8, {0, 3, 7});
    const SpMat A = tridiagonal(8, 1.0);
    const ReducedMatrix reduced(reduction, A);
    expect_same(reduced.matrix(), reduction.reduce(A));
    EXPECT_TRUE(reduced.matrix().isCompressed());
}

TEST(ReducedMatrix, AssignRefreshesValuesInPlace) {
    const DofReduction reduction(8, {0, 3, 7});
    ReducedMatrix reduced(reduction, tridiagonal(8, 1.0));
    const double* values = reduced.matrix().valuePtr();
    const int* inner = reduced.matrix().innerIndexPtr();

    const SpMat B = tridiagonal(8, 2.5);
    EXPECT_TRUE(reduced.has_pattern(B));
    reduced.assign(B);
    expect_same(reduced.matrix(), reduction.reduce(B));
    EXPECT_EQ(reduced.matrix().valuePtr(), values);
    EXPECT_EQ(reduced.matrix().innerIndexPtr(), inner);
}

TEST(ReducedMatrix, AssignRejectsAnotherPattern) {
    const DofReduction reduction(8, {0, 3, 7});
    ReducedMatrix reduced(reduction, tridiagonal(8, 1.0));
    const SpMat before = reduced.matrix();

    SpMat denser = tridiagonal(8, 1.0);
    denser.insert(0, 7) = 1.0;
    denser.insert(7, 0) = 1.0;
    denser.makeCompressed();
    EXPECT_FALSE(reduced.has_pattern(denser));
    EXPECT_THROW(reduced.assign(denser), std::invalid_argument);
    expect_same(reduced.matrix(), before);
}

TEST(ReducedMatrix, CoversNoAndAllFixedDofs) {
    const SpMat A = tridiagonal(5, 1.0);
    expect_same(ReducedMatrix(DofReduction(5, {}), A).matrix(), A);
    const ReducedMatrix none_free(DofReduction(5, {0, 1, 2, 3, 4}), A);
    EXPECT_EQ(none_free.matrix().rows(), 0);
    EXPECT_EQ(none_free.matrix().nonZeros(), 0);
}

TEST(ReducedMatrix, ChecksTheMatrix) {
    const DofReduction reduction(5, {1});
    EXPECT_THROW(ReducedMatrix(reduction, tridiagonal(4, 1.0)), std::invalid_argument);
    const SpMat A = tridiagonal(5, 1.);
    ReducedMatrix reduced(reduction, A);
    SpMat uncompressed = A;
    uncompressed.uncompress();
    EXPECT_FALSE(reduced.has_pattern(uncompressed));
    SpMat rectangular = A;
    rectangular.conservativeResize(6, 5);
    EXPECT_FALSE(reduced.has_pattern(rectangular));
    EXPECT_THROW(reduced.assign(rectangular), std::invalid_argument);
}

TEST(ReducedMatrix, StoresTheWholeDiagonal) {
    // Off-diagonal coupling only: A stores no diagonal at all.
    const std::vector<Triplet> entries = {{0, 2, 1.0}, {2, 0, 1.0}, {1, 3, -2.0}, {3, 1, -2.0}};
    SpMat A(4, 4);
    A.setFromTriplets(entries.begin(), entries.end());
    const DofReduction reduction(4, {3});
    const ReducedMatrix reduced(reduction, A);
    ASSERT_EQ(reduced.matrix().nonZeros(), 5); // (0,2), (2,0) and three diagonal zeros
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(reduced.matrix().coeff(i, i), 0.0);
    }
    EXPECT_EQ(reduced.matrix().coeff(0, 2), 1.0);
    EXPECT_TRUE(linalg::is_canonical(reduced.matrix()));
}

TEST(ReducedMatrix, ShiftsTheDiagonalUntilTheNextAssign) {
    const DofReduction reduction(6, {2});
    const SpMat A = tridiagonal(6, 1.0);
    ReducedMatrix reduced(reduction, A);
    const Eigen::MatrixXd before = Eigen::MatrixXd(reduced.matrix());
    reduced.add_to_diagonal(0.5);
    EXPECT_EQ(Eigen::MatrixXd(reduced.matrix()), before + 0.5 * Eigen::MatrixXd::Identity(5, 5));
    reduced.assign(A);
    EXPECT_EQ(Eigen::MatrixXd(reduced.matrix()), before);
}

} // namespace litepgo::dof::test
