#include "litepgo/linalg/sparse_pattern.hpp"

#include <gtest/gtest.h>

#include <utility>
#include <vector>

namespace litepgo::linalg::test {

namespace {

SpMat from_triplets(int n, const std::vector<Triplet>& entries) {
    SpMat A(n, n);
    A.setFromTriplets(entries.begin(), entries.end());
    return A;
}

} // namespace

TEST(SparsePattern, IsCanonical) {
    SpMat A = from_triplets(3, {{0, 0, 1.0}, {2, 0, 1.0}, {1, 1, 1.0}});
    EXPECT_TRUE(litepgo::linalg::is_canonical(A));
    SpMat unsorted = A;
    std::swap(unsorted.innerIndexPtr()[0], unsorted.innerIndexPtr()[1]);
    EXPECT_FALSE(litepgo::linalg::is_canonical(unsorted));
    SpMat uncompressed(2, 2);
    uncompressed.reserve(Eigen::VectorXi::Constant(2, 1));
    uncompressed.insert(0, 0) = 1.0;
    EXPECT_FALSE(litepgo::linalg::is_canonical(uncompressed));
}

TEST(SparsePattern, SamePatternIgnoresValues) {
    const SpMat A = from_triplets(3, {{0, 0, 1.0}, {1, 2, 5.0}});
    const SpMat B = from_triplets(3, {{0, 0, -4.0}, {1, 2, 0.0}});
    const SpMat C = from_triplets(3, {{0, 0, 1.0}, {2, 1, 5.0}});
    EXPECT_TRUE(litepgo::linalg::same_pattern(A, B));
    EXPECT_FALSE(litepgo::linalg::same_pattern(A, C));
    EXPECT_FALSE(litepgo::linalg::same_pattern(A, from_triplets(4, {{0, 0, 1.0}, {1, 2, 5.0}})));
}

TEST(SparsePattern, PatternOfZeroesValues) {
    const SpMat A = from_triplets(3, {{0, 0, 1.0}, {1, 2, 5.0}});
    const SpMat pattern = litepgo::linalg::pattern_of(A);
    EXPECT_TRUE(litepgo::linalg::same_pattern(pattern, A));
    EXPECT_EQ(pattern.coeff(0, 0), 0.0);
    EXPECT_EQ(pattern.coeff(1, 2), 0.0);
}

} // namespace litepgo::linalg::test
