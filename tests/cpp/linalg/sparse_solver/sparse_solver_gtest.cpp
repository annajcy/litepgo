#include "litepgo/linalg/sparse_solver/auto_sparse_solver.hpp"
#include "litepgo/linalg/sparse_solver/eigen_ldlt.hpp"
#include "litepgo/linalg/sparse_solver/eigen_sparse_lu.hpp"
#include "litepgo/linalg/sparse_solver/sparse_solver.hpp"

#if defined(LITEPGO_HAS_ACCELERATE)
#include "litepgo/linalg/sparse_solver/accelerate_ldlt.hpp"
#include "litepgo/linalg/sparse_solver/accelerate_llt.hpp"
#endif
#if defined(LITEPGO_HAS_MKL)
#include "litepgo/linalg/sparse_solver/pardiso_ldlt.hpp"
#include "litepgo/linalg/sparse_solver/pardiso_llt.hpp"
#endif

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace litepgo::linalg::test {

namespace {

// 1D Laplacian with Dirichlet ends, scaled: symmetric positive definite.
SpMat laplacian(int n, double scale = 1.0) {
    std::vector<Triplet> entries;
    for (int i = 0; i < n; ++i) {
        entries.emplace_back(i, i, 2.0 * scale);
        if (i + 1 < n) {
            entries.emplace_back(i, i + 1, -scale);
            entries.emplace_back(i + 1, i, -scale);
        }
    }
    SpMat A(n, n);
    A.setFromTriplets(entries.begin(), entries.end());
    return A;
}

// Saddle-point matrix [[K, B^T], [B, 0]] with K a Laplacian and B a row of ones:
// symmetric, nonsingular, indefinite. By default the zero diagonal entry is stored,
// so a diagonal shift stays within the pattern; scipy.sparse.bmat leaves it out.
SpMat saddle_point(int n, bool store_zero_diagonal = true) {
    std::vector<Triplet> entries;
    const SpMat K = laplacian(n);
    for (int column = 0; column < K.outerSize(); ++column) {
        for (SpMat::InnerIterator it(K, column); it; ++it) {
            entries.emplace_back(it.row(), it.col(), it.value());
        }
    }
    for (int i = 0; i < n; ++i) {
        entries.emplace_back(n, i, 1.0);
        entries.emplace_back(i, n, 1.0);
    }
    if (store_zero_diagonal) {
        entries.emplace_back(n, n, 0.0);
    }
    SpMat A(n + 1, n + 1);
    A.setFromTriplets(entries.begin(), entries.end());
    return A;
}

Vec dense_solution(const SpMat& A, const Vec& rhs) {
    return Eigen::MatrixXd(A).partialPivLu().solve(rhs);
}

// Checks the solve against a dense LU, given a solver already factored for A.
void expect_solution(const litepgo::linalg::SparseSolver& solver, const SpMat& A) {
    const Vec rhs = Vec::LinSpaced(A.rows(), -1.0, 2.0);
    EXPECT_LT((solver.solve(rhs) - dense_solution(A, rhs)).norm(), 1e-10);
}

// The pattern of laplacian(4), with the off-diagonals moved: same size and nnz.
SpMat laplacian4_lookalike() {
    const std::vector<Triplet> entries = {{0, 0, 4.0}, {1, 1, 4.0}, {2, 2, 4.0}, {3, 3, 4.0}, {0, 2, 1.0},
                                          {2, 0, 1.0}, {1, 3, 1.0}, {3, 1, 1.0}, {0, 3, 1.0}, {3, 0, 1.0}};
    SpMat A(4, 4);
    A.setFromTriplets(entries.begin(), entries.end());
    return A;
}

// A test backend over Eigen's LDLT that counts analyses and can be told to fail.
class CountingSolver final : public litepgo::linalg::SparseSolver {
  public:
    int analyze_calls = 0;
    bool fail_next_factor = false;
    bool throw_in_factor = false;
    bool solve_inside_factor = false;

  private:
    bool analyze_pattern(const SpMat& A) override {
        ++analyze_calls;
        solver_.analyzePattern(A);
        return true;
    }
    bool factor_values(const SpMat& A) override {
        if (solve_inside_factor) {
            solve(Vec::Zero(A.rows()));
        }
        if (throw_in_factor) {
            throw std::runtime_error("backend failure");
        }
        if (fail_next_factor) {
            fail_next_factor = false;
            return false;
        }
        solver_.factorize(A);
        return solver_.info() == Eigen::Success;
    }
    void solve_factored(const litepgo::ConstVecRef& rhs, litepgo::VecRef x) const override {
        x = solver_.solve(rhs);
    }

    Eigen::SimplicialLDLT<SpMat> solver_;
};

} // namespace

// Every backend on a positive definite matrix.
template <class Solver> class AllSolvers : public ::testing::Test {};
using AllSolverTypes = ::testing::Types<litepgo::linalg::EigenLDLT, litepgo::linalg::EigenSparseLU
#if defined(LITEPGO_HAS_ACCELERATE)
                                        ,
                                        litepgo::linalg::AccelerateLLT, litepgo::linalg::AccelerateLDLT
#endif
#if defined(LITEPGO_HAS_MKL)
                                        ,
                                        litepgo::linalg::PardisoLLT, litepgo::linalg::PardisoLDLT
#endif
                                        >;
TYPED_TEST_SUITE(AllSolvers, AllSolverTypes);

TYPED_TEST(AllSolvers, ComputesAndSolves) {
    TypeParam solver;
    const SpMat A = laplacian(40);
    ASSERT_TRUE(solver.compute(A));
    expect_solution(solver, A);
}

TYPED_TEST(AllSolvers, RefactorsNewValuesWithoutReanalyzing) {
    TypeParam solver;
    ASSERT_TRUE(solver.analyze(laplacian(40)));
    for (const double scale : {1.0, 3.5, 0.25}) {
        const SpMat A = laplacian(40, scale);
        ASSERT_TRUE(solver.factorize(A));
        expect_solution(solver, A);
    }
}

TYPED_TEST(AllSolvers, AnalyzesANewPattern) {
    TypeParam solver;
    ASSERT_TRUE(solver.compute(laplacian(40)));
    const SpMat B = laplacian(25);
    ASSERT_TRUE(solver.analyze(B));
    ASSERT_TRUE(solver.factorize(B));
    expect_solution(solver, B);
}

TYPED_TEST(AllSolvers, SolvesIntoProvidedVector) {
    TypeParam solver;
    const SpMat A = laplacian(10);
    const Vec rhs = Vec::Ones(10);
    ASSERT_TRUE(solver.compute(A));
    Vec x = Vec::Zero(10);
    solver.solve(rhs, x);
    EXPECT_LT((x - dense_solution(A, rhs)).norm(), 1e-10);
}

// Backends that handle symmetric indefinite matrices.
template <class Solver> class IndefiniteSolvers : public ::testing::Test {};
using IndefiniteSolverTypes = ::testing::Types<litepgo::linalg::EigenSparseLU
#if defined(LITEPGO_HAS_ACCELERATE)
                                               ,
                                               litepgo::linalg::AccelerateLDLT
#endif
#if defined(LITEPGO_HAS_MKL)
                                               ,
                                               litepgo::linalg::PardisoLDLT
#endif
                                               >;
TYPED_TEST_SUITE(IndefiniteSolvers, IndefiniteSolverTypes);

TYPED_TEST(IndefiniteSolvers, SolvesSaddlePoint) {
    TypeParam solver;
    const SpMat A = saddle_point(12);
    ASSERT_TRUE(solver.compute(A));
    expect_solution(solver, A);
}

// PARDISO reads out of bounds, only now and then, on a symmetric matrix that does
// not store its whole diagonal; repeat to give a regression a chance to show.
TYPED_TEST(IndefiniteSolvers, SolvesSaddlePointWithoutItsZeroDiagonal) {
    const SpMat A = saddle_point(12, false);
    ASSERT_EQ(A.nonZeros() + 1, saddle_point(12).nonZeros());
    for (int repeat = 0; repeat < 200; ++repeat) {
        TypeParam solver;
        ASSERT_TRUE(solver.compute(A));
        expect_solution(solver, A);
    }
}

// Cholesky backends reject an indefinite matrix but keep its analysis.
template <class Solver> void expect_cholesky_fails_on_indefinite() {
    Solver solver;
    const SpMat A = saddle_point(12);
    ASSERT_TRUE(solver.analyze(A));
    EXPECT_FALSE(solver.factorize(A));
    EXPECT_THROW(solver.solve(Vec::Ones(13)), std::logic_error);

    // Same pattern, shifted to positive definite: factorize again without analyze.
    SpMat shifted = A;
    shifted.diagonal().array() += 20.0;
    ASSERT_TRUE(solver.factorize(shifted));
    expect_solution(solver, shifted);
}

#if defined(LITEPGO_HAS_ACCELERATE)
TEST(AccelerateLLT, FailsOnIndefiniteButKeepsTheAnalysis) {
    expect_cholesky_fails_on_indefinite<litepgo::linalg::AccelerateLLT>();
}
#endif

#if defined(LITEPGO_HAS_MKL)
TEST(PardisoLLT, FailsOnIndefiniteButKeepsTheAnalysis) {
    expect_cholesky_fails_on_indefinite<litepgo::linalg::PardisoLLT>();
}

TEST(PardisoLLT, FailsOnIndefiniteWithoutItsZeroDiagonal) {
    const SpMat A = saddle_point(12, false);
    for (int repeat = 0; repeat < 200; ++repeat) {
        litepgo::linalg::PardisoLLT solver;
        EXPECT_FALSE(solver.compute(A));
    }
}
#endif

TEST(AutoSparseSolver, AliasesThePlatformDefaults) {
#if defined(LITEPGO_HAS_ACCELERATE)
    EXPECT_TRUE((std::is_same_v<litepgo::linalg::AutoSparseSolver, litepgo::linalg::AccelerateLLT>));
    EXPECT_TRUE((std::is_same_v<litepgo::linalg::AutoIndefiniteSparseSolver, litepgo::linalg::AccelerateLDLT>));
#elif defined(LITEPGO_HAS_MKL)
    EXPECT_TRUE((std::is_same_v<litepgo::linalg::AutoSparseSolver, litepgo::linalg::PardisoLLT>));
    EXPECT_TRUE((std::is_same_v<litepgo::linalg::AutoIndefiniteSparseSolver, litepgo::linalg::PardisoLDLT>));
#else
    EXPECT_TRUE((std::is_same_v<litepgo::linalg::AutoSparseSolver, litepgo::linalg::EigenLDLT>));
    EXPECT_TRUE((std::is_same_v<litepgo::linalg::AutoIndefiniteSparseSolver, litepgo::linalg::EigenSparseLU>));
#endif
}

TEST(SparseSolver, AnalyzesOnlyWhenAsked) {
    CountingSolver solver;
    ASSERT_TRUE(solver.analyze(laplacian(20)));
    ASSERT_TRUE(solver.factorize(laplacian(20)));
    ASSERT_TRUE(solver.factorize(laplacian(20, 2.0)));
    EXPECT_EQ(solver.analyze_calls, 1);
    ASSERT_TRUE(solver.compute(laplacian(20)));
    EXPECT_EQ(solver.analyze_calls, 2);
}

TEST(SparseSolver, FactorizeRequiresAnAnalysis) {
    litepgo::linalg::EigenLDLT solver;
    EXPECT_THROW(solver.factorize(laplacian(5)), std::logic_error);
}

TEST(SparseSolver, FactorizeRejectsAnotherPattern) {
    litepgo::linalg::EigenLDLT solver;
    ASSERT_TRUE(solver.analyze(laplacian(4)));
    const SpMat lookalike = laplacian4_lookalike();
    ASSERT_EQ(lookalike.nonZeros(), laplacian(4).nonZeros());
    EXPECT_THROW(solver.factorize(lookalike), std::invalid_argument);
    EXPECT_THROW(solver.factorize(laplacian(5)), std::invalid_argument);
    // The analysis survives the rejected calls.
    ASSERT_TRUE(solver.factorize(laplacian(4)));
    expect_solution(solver, laplacian(4));
}

TEST(SparseSolver, NumericalFailureKeepsTheAnalysis) {
    CountingSolver solver;
    ASSERT_TRUE(solver.compute(laplacian(20)));
    solver.fail_next_factor = true;
    EXPECT_FALSE(solver.factorize(laplacian(20)));
    EXPECT_THROW(solver.solve(Vec::Ones(20)), std::logic_error);
    ASSERT_TRUE(solver.factorize(laplacian(20)));
    EXPECT_EQ(solver.analyze_calls, 1);
}

TEST(SparseSolver, BackendExceptionDiscardsTheAnalysis) {
    CountingSolver solver;
    ASSERT_TRUE(solver.analyze(laplacian(20)));
    solver.throw_in_factor = true;
    EXPECT_THROW(solver.factorize(laplacian(20)), std::runtime_error);
    solver.throw_in_factor = false;
    EXPECT_THROW(solver.factorize(laplacian(20)), std::logic_error);
}

TEST(SparseSolver, AnalyzeDiscardsTheFactorization) {
    litepgo::linalg::EigenLDLT solver;
    ASSERT_TRUE(solver.compute(laplacian(6)));
    ASSERT_TRUE(solver.analyze(laplacian(6)));
    EXPECT_THROW(solver.solve(Vec::Ones(6)), std::logic_error);
}

TEST(SparseSolver, AnalyzeRejectsInvalidMatrices) {
    litepgo::linalg::EigenLDLT solver;
    EXPECT_THROW(solver.analyze(SpMat(0, 0)), std::invalid_argument);
    EXPECT_THROW(solver.analyze(SpMat(3, 4)), std::invalid_argument);

    SpMat uncompressed(3, 3);
    uncompressed.reserve(Eigen::VectorXi::Constant(3, 1));
    for (int i = 0; i < 3; ++i) {
        uncompressed.insert(i, i) = 1.0;
    }
    ASSERT_FALSE(uncompressed.isCompressed());
    EXPECT_THROW(solver.analyze(uncompressed), std::invalid_argument);

    // Eigen asserts when it copies such matrices, so edit the row indices in place.
    // Column 0 of laplacian(2) holds rows {0, 1}.
    SpMat duplicated = laplacian(2);
    duplicated.innerIndexPtr()[1] = 0;
    EXPECT_THROW(solver.analyze(duplicated), std::invalid_argument);
    SpMat unsorted = laplacian(2);
    std::swap(unsorted.innerIndexPtr()[0], unsorted.innerIndexPtr()[1]);
    EXPECT_THROW(solver.analyze(unsorted), std::invalid_argument);
}

TEST(SparseSolver, ChecksSolveArguments) {
    litepgo::linalg::EigenLDLT solver;
    EXPECT_THROW(solver.solve(Vec::Ones(5)), std::logic_error);
    ASSERT_TRUE(solver.compute(laplacian(5)));
    EXPECT_THROW(solver.solve(Vec::Ones(4)), std::invalid_argument);
    Vec x = Vec::Ones(5);
    EXPECT_THROW(solver.solve(x, x), std::invalid_argument);
}

TEST(SparseSolver, RejectsOverlappingUse) {
    CountingSolver solver;
    ASSERT_TRUE(solver.analyze(laplacian(5)));
    solver.solve_inside_factor = true;
    EXPECT_THROW(solver.factorize(laplacian(5)), std::logic_error);
    solver.solve_inside_factor = false;
    ASSERT_TRUE(solver.compute(laplacian(5)));
    expect_solution(solver, laplacian(5));
}

} // namespace litepgo::linalg::test
