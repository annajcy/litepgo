#include "litepgo/linalg/reduced_matrix.h"

#include "litepgo/linalg/dof_reduction.h"

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

namespace {

using litepgo::DofReduction;
using litepgo::ReducedMatrix;
using litepgo::SpMat;
using litepgo::Triplet;

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
}
