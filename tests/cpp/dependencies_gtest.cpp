// Smoke tests that each third-party dependency compiles, links, and runs.
#include "litepgo/core/build_info.h"

#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <gtest/gtest.h>
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_reduce.h>

#if defined(LITEPGO_HAS_ACCELERATE)
#include <Eigen/AccelerateSupport>
#endif

#include <functional>
#include <vector>

namespace {

// 1D Laplacian with Dirichlet ends: symmetric positive definite.
Eigen::SparseMatrix<double> laplacian(int n) {
    std::vector<Eigen::Triplet<double>> entries;
    for (int i = 0; i < n; ++i) {
        entries.emplace_back(i, i, 2.0);
        if (i + 1 < n) {
            entries.emplace_back(i, i + 1, -1.0);
            entries.emplace_back(i + 1, i, -1.0);
        }
    }
    Eigen::SparseMatrix<double> A(n, n);
    A.setFromTriplets(entries.begin(), entries.end());
    return A;
}

} // namespace

TEST(Dependencies, EigenSimplicialLDLTSolves) {
    const Eigen::SparseMatrix<double> A = laplacian(50);
    const Eigen::VectorXd expected = Eigen::VectorXd::LinSpaced(50, -1.0, 1.0);
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(A);
    ASSERT_EQ(solver.info(), Eigen::Success);
    EXPECT_LT((solver.solve(A * expected) - expected).norm(), 1e-10);
}

TEST(Dependencies, TbbParallelReduceSums) {
    const int n = 100000;
    const long long sum = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, n), 0LL,
        [](const tbb::blocked_range<int>& range, long long partial) {
            for (int i = range.begin(); i != range.end(); ++i) {
                partial += i;
            }
            return partial;
        },
        std::plus<long long>());
    EXPECT_EQ(sum, static_cast<long long>(n) * (n - 1) / 2);
}

TEST(Dependencies, BuildInfoListsBackends) {
    const litepgo::BuildInfo info = litepgo::build_info();
    EXPECT_EQ(info.eigen_version, "5.0.1");
    EXPECT_FALSE(info.tbb_runtime_version.empty());
    ASSERT_FALSE(info.sparse_backends.empty());
    EXPECT_EQ(info.sparse_backends.front(), "EigenLDLT");
}

#if defined(LITEPGO_HAS_ACCELERATE)
TEST(Dependencies, AccelerateLDLTSolves) {
    const Eigen::SparseMatrix<double> A = laplacian(50);
    const Eigen::VectorXd expected = Eigen::VectorXd::LinSpaced(50, -1.0, 1.0);
    // Eigen's Accelerate wrapper reads the right-hand side through data(), so it must be a concrete vector.
    const Eigen::VectorXd rhs = A * expected;
    Eigen::AccelerateLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);
    ASSERT_EQ(solver.info(), Eigen::Success);
    EXPECT_LT((solver.solve(rhs) - expected).norm(), 1e-10);
}
#endif
