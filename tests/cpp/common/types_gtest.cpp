#include "litepgo/common/types.hpp"

#include <gtest/gtest.h>

#include <type_traits>
#include <utility>
#include <vector>

namespace litepgo::test {

namespace {

static_assert(!litepgo::SpMat::IsRowMajor, "SpMat must be column-major");
static_assert(std::is_same_v<litepgo::SpMat::StorageIndex, int>, "SpMat indices must be int");
static_assert(
    std::is_same_v<std::remove_cvref_t<decltype(std::declval<litepgo::Triplet>().row())>, litepgo::SpMat::StorageIndex>,
    "Triplet indices must match SpMat");

double sum(const litepgo::ConstVecRef& x) {
    return x.sum();
}

const double* data_of(const litepgo::ConstVecRef& x) {
    return x.data();
}

void fill(litepgo::VecRef x, double value) {
    x.setConstant(value);
}

} // namespace

TEST(Types, ConstVecRefBindsWithoutCopying) {
    litepgo::Vec x = litepgo::Vec::LinSpaced(6, 0.0, 5.0);
    EXPECT_EQ(sum(x), 15.0);
    EXPECT_EQ(data_of(x), x.data());
    EXPECT_EQ(data_of(x.segment(2, 3)), x.data() + 2);

    const std::vector<double> raw = {1.0, 2.0, 3.0};
    const Eigen::Map<const litepgo::Vec> mapped(raw.data(), 3);
    EXPECT_EQ(data_of(mapped), raw.data());
}

TEST(Types, VecRefWritesThrough) {
    litepgo::Vec x = litepgo::Vec::Zero(5);
    fill(x.segment(1, 3), 2.0);
    EXPECT_EQ(x, (litepgo::Vec(5) << 0.0, 2.0, 2.0, 2.0, 0.0).finished());
}

TEST(Types, SymmetricSpMatStoresBothTriangles) {
    // [[4, 1, 0], [1, 3, 2], [0, 2, 5]], built from both triangles.
    const std::vector<litepgo::Triplet> entries = {{0, 0, 4.0}, {0, 1, 1.0}, {1, 0, 1.0}, {1, 1, 3.0},
                                                   {1, 2, 2.0}, {2, 1, 2.0}, {2, 2, 5.0}};
    litepgo::SpMat A(3, 3);
    A.setFromTriplets(entries.begin(), entries.end());

    EXPECT_EQ(A.nonZeros(), 7);
    EXPECT_TRUE(A.isCompressed());
    EXPECT_EQ(Eigen::MatrixXd(A), Eigen::MatrixXd(A.transpose()));

    const litepgo::Vec x(Eigen::Vector3d(1.0, -1.0, 2.0));
    const litepgo::Vec expected = Eigen::MatrixXd(A) * x;
    EXPECT_EQ(litepgo::Vec(A * x), expected);
}

} // namespace litepgo::test
