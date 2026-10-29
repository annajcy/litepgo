#if defined(LITEPGO_HAS_MKL)

#include "litepgo/linalg/sparse_solver/pardiso_ldlt.hpp"
#include "litepgo/linalg/sparse_solver/pardiso_llt.hpp"

#include <dlfcn.h>
#include <gtest/gtest.h>
#include <link.h>
#include <oneapi/tbb/global_control.h>
#include <oneapi/tbb/info.h>
#include <oneapi/tbb/task_arena.h>
#include <oneapi/tbb/version.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace litepgo::linalg::test {

namespace {

// 7-point Laplacian on an n^3 grid: symmetric positive definite, with enough fill
// that PARDISO factors it in parallel.
SpMat laplacian_3d(int n) {
    const auto index = [n](int i, int j, int k) { return (i * n + j) * n + k; };
    std::vector<Triplet> entries;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                const int row = index(i, j, k);
                entries.emplace_back(row, row, 6.0);
                for (const int neighbor : {i + 1 < n ? index(i + 1, j, k) : -1, j + 1 < n ? index(i, j + 1, k) : -1,
                                           k + 1 < n ? index(i, j, k + 1) : -1}) {
                    if (neighbor >= 0) {
                        entries.emplace_back(row, neighbor, -1.0);
                        entries.emplace_back(neighbor, row, -1.0);
                    }
                }
            }
        }
    }
    SpMat A(n * n * n, n * n * n);
    A.setFromTriplets(entries.begin(), entries.end());
    return A;
}

// User plus system CPU time, in clock ticks, of every thread of this process.
std::map<int, long> thread_cpu_ticks() {
    std::map<int, long> ticks;
    for (const auto& entry : std::filesystem::directory_iterator("/proc/self/task")) {
        std::ifstream file(entry.path() / "stat");
        std::string stat;
        std::getline(file, stat);
        // Fields after the parenthesized command name start at field 3 (state);
        // utime and stime are fields 14 and 15.
        std::istringstream fields(stat.substr(stat.rfind(')') + 2));
        std::string skipped;
        for (int field = 3; field < 14; ++field) {
            fields >> skipped;
        }
        long utime = 0;
        long stime = 0;
        fields >> utime >> stime;
        ticks[std::stoi(entry.path().filename().string())] = utime + stime;
    }
    return ticks;
}

// Number of threads that used CPU while work ran. Counted from the kernel's
// per-thread times, so it sees every thread, whichever arena or pool it belongs to.
template <class Work> int busy_threads(Work&& work) {
    // Let workers spinning after earlier parallel work go to sleep.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const std::map<int, long> before = thread_cpu_ticks();
    work();
    int busy = 0;
    for (const auto& [thread, ticks] : thread_cpu_ticks()) {
        const auto previous = before.find(thread);
        busy += ticks > (previous == before.end() ? 0 : previous->second);
    }
    return busy;
}

void expect_small_residual(const SparseSolver& solver, const SpMat& A) {
    const Vec rhs = Vec::LinSpaced(A.rows(), -1.0, 2.0);
    EXPECT_LT((A * solver.solve(rhs) - rhs).norm(), 1e-10 * rhs.norm());
}

} // namespace

template <class Solver> class PardisoThreading : public ::testing::Test {};
using PardisoTypes = ::testing::Types<PardisoLLT, PardisoLDLT>;
TYPED_TEST_SUITE(PardisoThreading, PardisoTypes);

TYPED_TEST(PardisoThreading, StaysWithinTheCallersTaskArena) {
    const SpMat A = laplacian_3d(32);
    for (const int threads : {1, 4}) {
        TypeParam solver;
        tbb::task_arena arena(threads);
        const int busy = busy_threads([&] { arena.execute([&] { ASSERT_TRUE(solver.compute(A)); }); });
        EXPECT_LE(busy, threads) << "task_arena(" << threads << ")";
        expect_small_residual(solver, A);
    }
}

TYPED_TEST(PardisoThreading, StaysWithinGlobalControl) {
    const SpMat A = laplacian_3d(32);
    for (const int threads : {1, 4}) {
        TypeParam solver;
        tbb::global_control limit(tbb::global_control::max_allowed_parallelism, threads);
        const int busy = busy_threads([&] { ASSERT_TRUE(solver.compute(A)); });
        EXPECT_LE(busy, threads) << "max_allowed_parallelism " << threads;
        expect_small_residual(solver, A);
    }
}

TYPED_TEST(PardisoThreading, FactorsInParallelWhenAllowed) {
    if (tbb::info::default_concurrency() < 2) {
        GTEST_SKIP() << "needs at least two CPUs";
    }
    const SpMat A = laplacian_3d(32);
    TypeParam solver;
    tbb::task_arena arena(4);
    const int busy = busy_threads([&] { arena.execute([&] { ASSERT_TRUE(solver.compute(A)); }); });
    EXPECT_GT(busy, 1);
    expect_small_residual(solver, A);
}

// MKL's threading layer must bind to the TBB runtime litepgo uses, not a second
// copy such as a system libtbb, or tbb::global_control would not reach it.
TEST(PardisoThreading, SharesLitepgoTbbRuntime) {
    std::vector<std::string> tbb_runtimes;
    dl_iterate_phdr(
        [](dl_phdr_info* info, size_t, void* data) {
            const std::filesystem::path path = info->dlpi_name;
            if (path.filename().string().starts_with("libtbb.so")) {
                static_cast<std::vector<std::string>*>(data)->push_back(info->dlpi_name);
            }
            return 0;
        },
        &tbb_runtimes);
    ASSERT_EQ(tbb_runtimes.size(), 1u) << ::testing::PrintToString(tbb_runtimes);

    Dl_info litepgo_tbb{};
    ASSERT_NE(dladdr(reinterpret_cast<void*>(&TBB_runtime_version), &litepgo_tbb), 0);
    EXPECT_EQ(std::filesystem::canonical(litepgo_tbb.dli_fname), std::filesystem::canonical(tbb_runtimes[0]));
    void* threading_layer = dlopen("libmkl_tbb_thread.so.3", RTLD_NOW | RTLD_NOLOAD);
    EXPECT_NE(threading_layer, nullptr) << "MKL is not on its TBB threading layer";
    if (threading_layer != nullptr) {
        dlclose(threading_layer);
    }
}

} // namespace litepgo::linalg::test

#endif
