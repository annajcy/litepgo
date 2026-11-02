#include "../../testing/element_formula_response.hpp"
#include "litepgo/dof/stencil_dof_map.hpp"

#include "litepgo/dof/constant_dof_map.hpp"
#include "litepgo/dof/element_dof_map.hpp"
#include "litepgo/dof/vertex_dof_map.hpp"
#include "litepgo/energy/response/mesh_response.hpp"
#include <atomic>
#include <gtest/gtest.h>

using namespace litepgo;
using litepgo::testing::formula_response;
namespace {
const mesh::TetTopology topology(5, {{0, 1, 2, 3}, {1, 2, 3, 4}});
Eigen::Matrix<double, 4, 2> local_matrix() {
    Eigen::Matrix<double, 4, 2> A;
    A << 1, 2, 3, 4, 5, 6, 7, 8;
    return A;
}
} // namespace
TEST(MeshResponse, SharedParametersAndOutputsAccumulate) {
    int derivatives = 0;
    auto r = formula_response(dof::ConstantDofMap(topology, 2), dof::VertexDofMap(topology, 1),
                              [&](int e, const Vec& x, Eigen::MatrixXd* J) -> Vec {
                                  const Eigen::MatrixXd A = (e + 1) * local_matrix();
                                  if (J) {
                                      ++derivatives;
                                      *J = A;
                                  }
                                  return A * x;
                              });
    Eigen::Matrix<double, 5, 2> expected;
    expected << 1, 2, 5, 8, 11, 14, 17, 20, 14, 16;
    const Vec x = Eigen::Vector2d(2, 3);
    EXPECT_EQ(r.num_inputs(), 2);
    EXPECT_EQ(r.num_outputs(), 5);
    EXPECT_TRUE(r.has_fixed_pattern());
    EXPECT_FALSE(r.update_pattern(x));
    EXPECT_TRUE(r.eval_value(x).isApprox(expected * x));
    EXPECT_EQ(derivatives, 0);
    SpMat J = r.jacobian_pattern();
    EXPECT_TRUE(r.eval(x, &J).isApprox(expected * x));
    EXPECT_TRUE(Eigen::MatrixXd(J).isApprox(expected));
    EXPECT_EQ(derivatives, 2);
    EXPECT_TRUE(r.eval(-x, &J).isApprox(-expected * x));
    EXPECT_TRUE(Eigen::MatrixXd(J).isApprox(expected));
    EXPECT_EQ(J.nonZeros(), 10);
}
TEST(MeshResponse, IndependentParameterBlocks) {
    auto r = formula_response(dof::ElementDofMap(topology, 2), dof::VertexDofMap(topology, 1),
                              [](int, const Vec& x, Eigen::MatrixXd* J) -> Vec {
                                  if (J)
                                      *J = local_matrix();
                                  return local_matrix() * x;
                              });
    Eigen::Matrix<double, 5, 4> expected;
    expected << 1, 2, 0, 0, 3, 4, 1, 2, 5, 6, 3, 4, 7, 8, 5, 6, 0, 0, 7, 8;
    Vec x(4);
    x << 1, 2, 3, 4;
    EXPECT_TRUE(Eigen::MatrixXd(r.eval_jacobian(x)).isApprox(expected));
    EXPECT_TRUE(r.eval_value(x).isApprox(expected * x));
}
TEST(MeshResponse, MatrixGatherPreservesRowMajorSlotsAndZeros) {
    auto r = formula_response(dof::VertexDofMap(topology, 2), dof::ElementDofMap(topology, 1),
                              [](int, const Eigen::MatrixXd& x, Eigen::MatrixXd* J) -> Vec {
                                  EXPECT_EQ(x.rows(), 4);
                                  EXPECT_EQ(x.cols(), 2);
                                  if (J) {
                                      J->setZero(1, 8);
                                      (*J)(0, 3) = 2 * x(1, 1);
                                  }
                                  return Vec::Constant(1, x(1, 1) * x(1, 1));
                              });
    const Vec x = Vec::LinSpaced(10, 1, 10);
    SpMat J = r.jacobian_pattern();
    EXPECT_EQ(r.eval(x, &J)[0], 16);
    EXPECT_EQ(J.coeff(0, 3), 8);
    EXPECT_EQ(J.coeff(1, 5), 12);
    r.eval(Vec::Zero(10), &J);
    EXPECT_EQ(J.nonZeros(), 16);
    EXPECT_TRUE(J.coeffs().isZero());
}
TEST(MeshResponse, RejectsInvalidLocalResultsAndCountMismatch) {
    const dof::ConstantDofMap input(topology, 2);
    const dof::VertexDofMap output(topology, 1);
    auto wrong = formula_response(input, output, [](int, const Vec&, Eigen::MatrixXd*) { return Vec::Zero(3).eval(); });
    EXPECT_THROW(wrong.eval_value(Vec::Zero(2)), std::logic_error);
    auto bad_j = formula_response(input, output, [](int, const Vec&, Eigen::MatrixXd* J) {
        if (J)
            J->setZero(2, 4);
        return Vec::Zero(4).eval();
    });
    EXPECT_THROW(bad_j.eval_jacobian(Vec::Zero(2)), std::logic_error);
    const mesh::TetTopology single(4, {{0, 1, 2, 3}});
    EXPECT_THROW((formula_response(input, dof::VertexDofMap(single, 1),
                                   [](int, const Vec&, Eigen::MatrixXd*) { return Vec::Zero(4).eval(); })),
                 std::invalid_argument);
}
TEST(MeshResponse, EmptyMeshAndAssemblyOverflow) {
    const mesh::TetTopology empty(0, {});
    auto zero =
        formula_response(dof::ConstantDofMap(empty, 2), dof::ElementDofMap(empty, 1),
                         [](int, const Vec&, Eigen::MatrixXd*) -> Vec { throw std::logic_error("Must not evaluate"); });
    EXPECT_EQ(zero.eval_value(Vec::Zero(2)).size(), 0);
    EXPECT_EQ(zero.eval_jacobian(Vec::Zero(2)).cols(), 2);
    auto huge = formula_response(
        dof::ConstantDofMap(topology, 2), dof::ConstantDofMap(topology, 1),
        [](int, const Vec&, Eigen::MatrixXd*) { return Vec::Constant(1, std::numeric_limits<double>::max()).eval(); });
    EXPECT_THROW(huge.eval_value(Vec::Zero(2)), std::domain_error);
}

TEST(MeshResponse, RejectsBadIndicesBeforeEvaluation) {
    struct Map {
        std::vector<int> indices;
        int num_elements() const {
            return 1;
        }
        int num_dofs() const {
            return 2;
        }
        std::vector<int> dof_indices(int) const {
            return indices;
        }
        Vec gather(int, const ConstVecRef& x) const {
            return x;
        }
    } valid{{0, 1}}, duplicate{{0, 0}}, outside{{0, 2}};
    auto evaluate = [](int, const Vec& x, Eigen::MatrixXd*) -> Vec { return x; };
    EXPECT_NO_THROW((formula_response(valid, duplicate, evaluate)));
    EXPECT_THROW((formula_response(outside, valid, evaluate)), std::invalid_argument);
}

TEST(MeshResponse, SharedAssemblyAndPartialBatch) {
    const mesh::TetTopology t(6, {{0, 1, 2, 3}, {1, 2, 3, 4}, {2, 3, 4, 5}});
    const dof::ConstantDofMap input(t, 2);
    const dof::VertexDofMap output(t, 1);
    auto regular = formula_response(input, output, [](int e, const Vec& x, Eigen::MatrixXd* J) -> Vec {
        if (J)
            *J = (e + 1) * local_matrix();
        return (e + 1) * local_matrix() * x;
    });
    for (int size : {1, 2, 16}) {
        std::atomic<int> calls = 0;
        energy::MeshResponse batch(
            input, output,
            [&](std::span<const int> ids, const energy::BatchMatrix& x, energy::BatchMatrix* J) {
                ++calls;
                energy::BatchMatrix values(ids.size(), 4);
                if (J)
                    J->resize(4 * ids.size(), 2);
                for (std::size_t b = 0; b < ids.size(); ++b) {
                    values.row(b) = ((ids[b] + 1) * local_matrix() * x.row(b).transpose()).transpose();
                    if (J)
                        J->middleRows(4 * b, 4) = (ids[b] + 1) * local_matrix();
                }
                return values;
            },
            size);
        const Vec x = Eigen::Vector2d(2, 3);
        SpMat J = batch.jacobian_pattern();
        EXPECT_TRUE(batch.eval(x, &J).isApprox(regular.eval_value(x)));
        EXPECT_TRUE(Eigen::MatrixXd(J).isApprox(Eigen::MatrixXd(regular.eval_jacobian(x))));
        EXPECT_EQ(calls.load(), (3 + size - 1) / size);
        EXPECT_TRUE(batch.eval_value(-x).isApprox(-regular.eval_value(x)));
    }
}

TEST(MeshResponse, ValidatesCallbackAndBatchResults) {
    const dof::ConstantDofMap input(topology, 2);
    const dof::VertexDofMap output(topology, 1);
    EXPECT_THROW((energy::MeshResponse(input, output, {})), std::invalid_argument);
    auto bad = [](std::span<const int>, const energy::BatchMatrix&, energy::BatchMatrix*) {
        return energy::BatchMatrix::Zero(1, 1).eval();
    };
    EXPECT_THROW((energy::MeshResponse(input, output, bad, 0)), std::invalid_argument);
    energy::MeshResponse batch(input, output, bad);
    EXPECT_THROW(batch.eval_value(Vec::Zero(2)), std::logic_error);
    EXPECT_THROW(batch.eval_jacobian(Vec::Zero(2)), std::logic_error);
}

TEST(MeshResponse, ReusedBuffersDetectIncompleteWritesAndRecover) {
    const dof::ConstantDofMap input(topology, 2);
    const dof::VertexDofMap output(topology, 1);
    bool omit = false;
    energy::MeshResponse batch(input, output,
                               [&](std::span<const int> ids, const energy::BatchMatrix& x, energy::BatchMatrix* J) {
                                   if (J && !omit)
                                       J->setOnes(4 * ids.size(), 2);
                                   return energy::BatchMatrix::Constant(ids.size(), 4, x.row(0).sum()).eval();
                               });
    const Vec x = Vec::Ones(2);
    const SpMat expected = batch.eval_jacobian(x);
    omit = true;
    EXPECT_THROW(batch.eval_jacobian(x), std::logic_error);
    omit = false;
    EXPECT_TRUE(Eigen::MatrixXd(batch.eval_jacobian(x)).isApprox(Eigen::MatrixXd(expected)));
    auto regular = formula_response(input, output, [&](int, const Vec& local, Eigen::MatrixXd* J) -> Vec {
        if (J && !omit)
            J->setOnes(4, 2);
        return Vec::Constant(4, local.sum());
    });
    EXPECT_TRUE(Eigen::MatrixXd(regular.eval_jacobian(x)).isApprox(Eigen::MatrixXd(expected)));
    omit = true;
    EXPECT_THROW(regular.eval_jacobian(x), std::logic_error);
    omit = false;
    EXPECT_TRUE(Eigen::MatrixXd(regular.eval_jacobian(x)).isApprox(Eigen::MatrixXd(expected)));
}

TEST(MeshResponse, EmptyLocalInputsAndOutputsRetainBatchDimensions) {
    const dof::StencilDofMap map(1, {{}, {}}, 1);
    energy::MeshResponse obj(
        map, map,
        [](auto ids, const auto& x, auto* J) {
            EXPECT_EQ(x.rows(), static_cast<Eigen::Index>(ids.size()));
            EXPECT_EQ(x.cols(), 0);
            if (J)
                J->resize(0, 0);
            return energy::BatchMatrix(ids.size(), 0);
        },
        1);
    EXPECT_TRUE(obj.eval_value(Vec::Zero(1)).isZero());
    EXPECT_EQ(obj.eval_jacobian(Vec::Zero(1)).nonZeros(), 0);
}

TEST(MeshResponse, DuplicateIndicesGatherAndScatterAcrossBatches) {
    struct RepeatedMap {
        int num_dofs() const {
            return 1;
        }
        int num_elements() const {
            return 3;
        }
        std::vector<int> dof_indices(int) const {
            return {0, 0};
        }
    } map;
    for (int size : {1, 2, 8}) {
        energy::MeshResponse obj(
            map, map,
            [](auto ids, const auto& x, auto* J) {
                energy::BatchMatrix values(ids.size(), 2);
                for (Eigen::Index b = 0; b < x.rows(); ++b) {
                    values.row(b).setConstant(x.row(b).sum());
                    if (J)
                        J->middleRows(2 * b, 2).setOnes();
                }
                return values;
            },
            size);
        for (double x : {2.0, -3.0}) {
            EXPECT_DOUBLE_EQ(obj.eval_value(Vec::Constant(1, x))[0], 12 * x);
            EXPECT_DOUBLE_EQ(obj.eval_jacobian(Vec::Constant(1, x)).coeff(0, 0), 12);
        }
    }
}
