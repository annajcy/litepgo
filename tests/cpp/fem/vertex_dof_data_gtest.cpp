#include "litepgo/fem/vertex_dof_data.hpp"
#include "litepgo/fem/vertex_element_dof_map.hpp"

#include <gtest/gtest.h>

#include <array>
#include <concepts>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace {

using litepgo::Vec;
using litepgo::VertexDofData;

template <int Width>
concept CanFormData = requires { typename VertexDofData<Width>; };

static_assert(CanFormData<1>);
static_assert(CanFormData<3>);
static_assert(!CanFormData<0>);
static_assert(!CanFormData<-1>);
static_assert(!std::is_convertible_v<int, VertexDofData<3>>);
static_assert(!std::is_convertible_v<Vec, VertexDofData<3>>);
static_assert(VertexDofData<3>::dofs_per_vertex() == 3);
static_assert(std::same_as<decltype(std::declval<VertexDofData<3>&>().values()), litepgo::VecRef>);
static_assert(std::same_as<decltype(std::declval<const VertexDofData<3>&>().values()), litepgo::ConstVecRef>);
static_assert(std::same_as<decltype(std::declval<VertexDofData<3>&>()(0, 0)), double&>);
static_assert(std::same_as<decltype(std::declval<const VertexDofData<3>&>()(0, 0)), double>);

} // namespace

TEST(VertexDofData, CreatesZeroDataWithMatchingCounts) {
    const VertexDofData<3> data(4);
    EXPECT_EQ(data.num_vertices(), 4);
    EXPECT_EQ(data.dofs_per_vertex(), 3);
    EXPECT_EQ(data.num_dofs(), 12);
    EXPECT_EQ(data.values().size(), 12);
    EXPECT_TRUE(data.values().isZero());
}

TEST(VertexDofData, NumbersConsecutiveVertexBlocks) {
    const VertexDofData<6> data(3);
    EXPECT_EQ(data.num_vertices(), 3);
    EXPECT_EQ(data.dofs_per_vertex(), 6);
    EXPECT_EQ(data.num_dofs(), 18);
    for (int vertex = 0; vertex < 3; ++vertex) {
        for (int slot = 0; slot < 6; ++slot) {
            EXPECT_EQ(data.dof(vertex, slot), 6 * vertex + slot);
        }
    }
}

TEST(VertexDofData, InfersVertexCountAndCopiesInputValues) {
    Vec values = Vec::LinSpaced(12, 1.0, 12.0);
    VertexDofData<3> data(values);
    EXPECT_EQ(data.num_vertices(), 4);
    EXPECT_EQ(data.values(), values);
    EXPECT_NE(data.values().data(), values.data());

    values[0] = -100.0;
    EXPECT_DOUBLE_EQ(data(0, 0), 1.0);
    data(1, 2) = 42.0;
    EXPECT_DOUBLE_EQ(values[5], 6.0);
    EXPECT_DOUBLE_EQ(data.values()[5], 42.0);
}

TEST(VertexDofData, CopiesEigenViewsAndTemporaryInputs) {
    const Vec padded = Vec::LinSpaced(10, 0.0, 9.0);
    const VertexDofData<3> segment(padded.segment(2, 6));
    const Vec expected = Vec::LinSpaced(6, 2.0, 7.0);
    EXPECT_EQ(segment.num_vertices(), 2);
    EXPECT_EQ(segment.values(), expected);

    std::array<double, 6> buffer{1, 2, 3, 4, 5, 6};
    const VertexDofData<3> mapped(Eigen::Map<const Vec>(buffer.data(), 6));
    buffer[0] = -1;
    EXPECT_DOUBLE_EQ(mapped(0, 0), 1.0);

    const VertexDofData<3> temporary(Vec::Constant(6, 3.5));
    EXPECT_TRUE(temporary.values().isConstant(3.5));
}

TEST(VertexDofData, MutableViewsAndVertexAccessReferToTheSameStorage) {
    VertexDofData<3> data(2);
    auto values = data.values();
    values.setConstant(2.0);
    values[4] = 7.0;
    EXPECT_DOUBLE_EQ(data(1, 1), 7.0);
    data(0, 2) = -3.0;
    EXPECT_DOUBLE_EQ(values[2], -3.0);

    const auto& constant = data;
    const auto read_only = constant.values();
    EXPECT_EQ(read_only.data(), values.data());
    EXPECT_DOUBLE_EQ(constant(0, 2), -3.0);
    EXPECT_EQ(data.num_dofs(), values.size());
}

TEST(VertexDofData, CopyMoveAndAssignmentKeepDataAndCountsConsistent) {
    VertexDofData<3> original(Vec::LinSpaced(6, 1.0, 6.0));
    VertexDofData<3> copied = original;
    original(0, 0) = 42.0;
    EXPECT_DOUBLE_EQ(copied(0, 0), 1.0);

    VertexDofData<3> moved = std::move(copied);
    EXPECT_DOUBLE_EQ(moved(0, 0), 1.0);
    EXPECT_EQ(moved.num_vertices(), 2);
    EXPECT_EQ(copied.num_dofs(), copied.values().size());
    EXPECT_EQ(copied.num_vertices() * copied.dofs_per_vertex(), copied.num_dofs());

    VertexDofData<3> assigned(1);
    assigned = original;
    EXPECT_EQ(assigned.num_vertices(), 2);
    assigned(0, 0) = -1.0;
    EXPECT_DOUBLE_EQ(original(0, 0), 42.0);

    VertexDofData<3> move_assigned(3);
    move_assigned = std::move(moved);
    EXPECT_EQ(move_assigned.num_vertices(), 2);
    EXPECT_DOUBLE_EQ(move_assigned(0, 0), 1.0);
    EXPECT_EQ(moved.num_dofs(), moved.values().size());
    EXPECT_EQ(moved.num_vertices() * moved.dofs_per_vertex(), moved.num_dofs());
}

TEST(VertexDofData, AllowsEmptyDataAndChecksVertexAndSlotBounds) {
    const VertexDofData<3> zero(0);
    VertexDofData<3> inferred{Vec(0)};
    EXPECT_EQ(zero.num_vertices(), 0);
    EXPECT_EQ(zero.num_dofs(), 0);
    EXPECT_EQ(inferred.num_vertices(), 0);
    EXPECT_THROW(zero.dof(0, 0), std::out_of_range);
    EXPECT_THROW(zero(0, 0), std::out_of_range);
    EXPECT_THROW(inferred(0, 0), std::out_of_range);

    VertexDofData<3> data(2);
    EXPECT_THROW(data.dof(-1, 0), std::out_of_range);
    EXPECT_THROW(data.dof(2, 0), std::out_of_range);
    EXPECT_THROW(data.dof(0, -1), std::out_of_range);
    EXPECT_THROW(data.dof(0, 3), std::out_of_range);
    EXPECT_THROW(data(-1, 0), std::out_of_range);
    EXPECT_THROW(data(2, 0), std::out_of_range);
    EXPECT_THROW(data(0, -1), std::out_of_range);
    EXPECT_THROW(data(0, 3), std::out_of_range);
    const auto& constant = data;
    EXPECT_THROW(constant(2, 0), std::out_of_range);
    EXPECT_THROW(constant(0, 3), std::out_of_range);
}

TEST(VertexDofData, RejectsInvalidCountsAndIncompleteVertexBlocks) {
    EXPECT_THROW(VertexDofData<3>(-1), std::invalid_argument);
    EXPECT_THROW(VertexDofData<3>(std::numeric_limits<int>::max() / 3 + 1), std::invalid_argument);
    EXPECT_THROW(VertexDofData<std::numeric_limits<int>::max()>(2), std::invalid_argument);
    const Vec five = Vec::Zero(5);
    EXPECT_THROW((VertexDofData<3>(five)), std::invalid_argument);
    const Vec seven = Vec::Zero(7);
    EXPECT_THROW((VertexDofData<2>(seven)), std::invalid_argument);
}

TEST(VertexDofData, ConnectsThreeIndependentFieldsToOneElementMap) {
    const litepgo::HexTopology topology(8, {{4, 2, 6, 0, 5, 3, 7, 1}});
    VertexDofData<3> displacement(8);
    VertexDofData<3> plastic_displacement(Vec::LinSpaced(24, -24.0, -1.0));
    VertexDofData<2> material_parameters(Vec::LinSpaced(16, 1.0, 16.0));
    displacement(4, 0) = 42.0;
    const litepgo::VertexElementDofMap map(topology);

    const auto a_e = map.gather(0, displacement);
    const auto b_e = map.gather(0, plastic_displacement);
    const auto theta_e = map.gather(0, material_parameters);
    EXPECT_DOUBLE_EQ(a_e(0, 0), 42.0);
    EXPECT_DOUBLE_EQ(b_e(0, 0), -12.0);
    EXPECT_DOUBLE_EQ(b_e(0, 1), -11.0);
    EXPECT_DOUBLE_EQ(b_e(0, 2), -10.0);
    EXPECT_DOUBLE_EQ(theta_e(0, 0), 9.0);
    EXPECT_DOUBLE_EQ(theta_e(0, 1), 10.0);
    EXPECT_DOUBLE_EQ(a_e(7, 2), displacement(1, 2));
    EXPECT_DOUBLE_EQ(b_e(7, 2), plastic_displacement(1, 2));
    EXPECT_DOUBLE_EQ(theta_e(7, 1), material_parameters(1, 1));
}
