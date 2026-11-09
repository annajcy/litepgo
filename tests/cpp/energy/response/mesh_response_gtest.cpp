#include "../../testing/batch_fixtures.hpp"
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
    EXPECT_TRUE(r.evaluate(x)->value().isApprox(expected * x));
    EXPECT_EQ(derivatives, 0);
    SpMat J;
    EXPECT_TRUE(([&] {
                    auto evaluation_0 = r.evaluate(x);
                    auto jacobian_0 = evaluation_0->jacobian();
                    auto result = std::pair{litepgo::Vec(evaluation_0->value()), std::move(jacobian_0)};
                    J = std::move(result.second);
                    return std::move(result.first);
                }())
                    .isApprox(expected * x));
    EXPECT_TRUE(Eigen::MatrixXd(J).isApprox(expected));
    EXPECT_EQ(derivatives, 2);
    EXPECT_TRUE(([&] {
                    auto evaluation_1 = r.evaluate(-x);
                    auto jacobian_1 = evaluation_1->jacobian();
                    auto result = std::pair{litepgo::Vec(evaluation_1->value()), std::move(jacobian_1)};
                    J = std::move(result.second);
                    return std::move(result.first);
                }())
                    .isApprox(-expected * x));
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
    EXPECT_TRUE(Eigen::MatrixXd(r.evaluate(x)->jacobian()).isApprox(expected));
    EXPECT_TRUE(r.evaluate(x)->value().isApprox(expected * x));
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
    SpMat J;
    EXPECT_EQ(([&] {
                  auto evaluation_2 = r.evaluate(x);
                  auto jacobian_2 = evaluation_2->jacobian();
                  auto result = std::pair{litepgo::Vec(evaluation_2->value()), std::move(jacobian_2)};
                  J = std::move(result.second);
                  return std::move(result.first);
              }())[0],
              16);
    EXPECT_EQ(J.coeff(0, 3), 8);
    EXPECT_EQ(J.coeff(1, 5), 12);
    ([&] {
        auto evaluation_3 = r.evaluate(Vec::Zero(10));
        auto jacobian_3 = evaluation_3->jacobian();
        auto result = std::pair{litepgo::Vec(evaluation_3->value()), std::move(jacobian_3)};
        J = std::move(result.second);
        return std::move(result.first);
    }());
    EXPECT_EQ(J.nonZeros(), 16);
    EXPECT_TRUE(J.coeffs().isZero());
}
TEST(MeshResponse, RejectsInvalidLocalResultsAndCountMismatch) {
    const dof::ConstantDofMap input(topology, 2);
    const dof::VertexDofMap output(topology, 1);
    auto wrong = formula_response(input, output, [](int, const Vec&, Eigen::MatrixXd*) { return Vec::Zero(3).eval(); });
    EXPECT_THROW(wrong.evaluate(Vec::Zero(2))->value(), std::logic_error);
    auto bad_j = formula_response(input, output, [](int, const Vec&, Eigen::MatrixXd* J) {
        if (J)
            J->setZero(2, 4);
        return Vec::Zero(4).eval();
    });
    EXPECT_THROW(bad_j.evaluate(Vec::Zero(2))->jacobian(), std::logic_error);
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
    EXPECT_EQ(zero.evaluate(Vec::Zero(2))->value().size(), 0);
    EXPECT_EQ(zero.evaluate(Vec::Zero(2))->jacobian().cols(), 2);
    auto huge = formula_response(
        dof::ConstantDofMap(topology, 2), dof::ConstantDofMap(topology, 1),
        [](int, const Vec&, Eigen::MatrixXd*) { return Vec::Constant(1, std::numeric_limits<double>::max()).eval(); });
    EXPECT_THROW(huge.evaluate(Vec::Zero(2))->value(), std::domain_error);
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
        ::litepgo::testing::BatchResponseFixture batch(
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
        SpMat J;
        EXPECT_TRUE(([&] {
                        auto evaluation_4 = batch.evaluate(x);
                        auto jacobian_4 = evaluation_4->jacobian();
                        auto result = std::pair{litepgo::Vec(evaluation_4->value()), std::move(jacobian_4)};
                        J = std::move(result.second);
                        return std::move(result.first);
                    }())
                        .isApprox(regular.evaluate(x)->value()));
        EXPECT_TRUE(Eigen::MatrixXd(J).isApprox(Eigen::MatrixXd(regular.evaluate(x)->jacobian())));
        EXPECT_EQ(calls.load(), (3 + size - 1) / size);
        EXPECT_TRUE(batch.evaluate(-x)->value().isApprox(-regular.evaluate(x)->value()));
    }
}

TEST(MeshResponse, ValidatesBatchSizeAndResults) {
    const dof::ConstantDofMap input(topology, 2);
    const dof::VertexDofMap output(topology, 1);
    auto bad = [](std::span<const int>, const energy::BatchMatrix&, energy::BatchMatrix*) {
        return energy::BatchMatrix::Zero(1, 1).eval();
    };
    EXPECT_THROW((::litepgo::testing::BatchResponseFixture(input, output, bad, 0)), std::invalid_argument);
    ::litepgo::testing::BatchResponseFixture batch(input, output, bad);
    EXPECT_THROW(batch.evaluate(Vec::Zero(2))->value(), std::logic_error);
    EXPECT_THROW(batch.evaluate(Vec::Zero(2))->jacobian(), std::logic_error);
}

TEST(MeshResponse, ReusedBuffersDetectIncompleteWritesAndRecover) {
    const dof::ConstantDofMap input(topology, 2);
    const dof::VertexDofMap output(topology, 1);
    bool omit = false;
    ::litepgo::testing::BatchResponseFixture batch(
        input, output, [&](std::span<const int> ids, const energy::BatchMatrix& x, energy::BatchMatrix* J) {
            if (J && !omit)
                J->setOnes(4 * ids.size(), 2);
            return energy::BatchMatrix::Constant(ids.size(), 4, x.row(0).sum()).eval();
        });
    const Vec x = Vec::Ones(2);
    const SpMat expected = batch.evaluate(x)->jacobian();
    omit = true;
    EXPECT_THROW(batch.evaluate(x)->jacobian(), std::logic_error);
    omit = false;
    EXPECT_TRUE(Eigen::MatrixXd(batch.evaluate(x)->jacobian()).isApprox(Eigen::MatrixXd(expected)));
    auto regular = formula_response(input, output, [&](int, const Vec& local, Eigen::MatrixXd* J) -> Vec {
        if (J && !omit)
            J->setOnes(4, 2);
        return Vec::Constant(4, local.sum());
    });
    EXPECT_TRUE(Eigen::MatrixXd(regular.evaluate(x)->jacobian()).isApprox(Eigen::MatrixXd(expected)));
    omit = true;
    EXPECT_THROW(regular.evaluate(x)->jacobian(), std::logic_error);
    omit = false;
    EXPECT_TRUE(Eigen::MatrixXd(regular.evaluate(x)->jacobian()).isApprox(Eigen::MatrixXd(expected)));
}

TEST(MeshResponse, EmptyLocalInputsAndOutputsRetainBatchDimensions) {
    const dof::StencilDofMap map(1, {{}, {}}, 1);
    ::litepgo::testing::BatchResponseFixture obj(
        map, map,
        [](auto ids, const auto& x, auto* J) {
            EXPECT_EQ(x.rows(), static_cast<Eigen::Index>(ids.size()));
            EXPECT_EQ(x.cols(), 0);
            if (J)
                J->resize(0, 0);
            return energy::BatchMatrix(ids.size(), 0);
        },
        1);
    EXPECT_TRUE(obj.evaluate(Vec::Zero(1))->value().isZero());
    EXPECT_EQ(obj.evaluate(Vec::Zero(1))->jacobian().nonZeros(), 0);
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
        ::litepgo::testing::BatchResponseFixture obj(
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
            EXPECT_DOUBLE_EQ(obj.evaluate(Vec::Constant(1, x))->value()[0], 12 * x);
            EXPECT_DOUBLE_EQ(obj.evaluate(Vec::Constant(1, x))->jacobian().coeff(0, 0), 12);
            EXPECT_DOUBLE_EQ(obj.evaluate(Vec::Constant(1, x))->jvp(Vec::Constant(1, 2))[0], 24);
            EXPECT_DOUBLE_EQ(obj.evaluate(Vec::Constant(1, x))->vjp(Vec::Constant(1, 3))[0], 36);
        }
    }
}

TEST(MeshResponse, ProductsAndDirectOverrides) {
    struct Map {
        int size;
        std::vector<std::vector<int>> indices;
        int num_dofs() const {
            return size;
        }
        int num_elements() const {
            return static_cast<int>(indices.size());
        }
        std::vector<int> dof_indices(int e) const {
            return indices.at(e);
        }
    };
    const Map input{2, {{0, 0}, {0, 1}}};
    const Map output{3, {{0, 1}, {1, 2}}};
    class Direct final : public energy::MeshResponse {
      public:
        Direct(const Map& i, const Map& o) : MeshResponse(i, o, 1) {}
        bool fail = false;

      private:
        energy::BatchMatrix eval_batch(std::span<const int>, const energy::BatchMatrix&,
                                       energy::BatchMatrix*) const override {
            throw std::logic_error("explicit Jacobian must not be requested");
        }
        energy::BatchMatrix eval_jvp_batch(std::span<const int>, const energy::BatchMatrix&,
                                           const energy::BatchMatrix& d) const override {
            if (fail)
                throw std::domain_error("test failure");
            return 2 * d;
        }
        energy::BatchMatrix eval_vjp_batch(std::span<const int>, const energy::BatchMatrix&,
                                           const energy::BatchMatrix& d) const override {
            return 2 * d;
        }
    };
    ::litepgo::testing::BatchResponseFixture fallback(
        input, output, [](std::span<const int>, const energy::BatchMatrix& x, energy::BatchMatrix* J) {
            if (J) {
                J->setZero(2 * x.rows(), 2);
                for (Eigen::Index b = 0; b < x.rows(); ++b)
                    J->middleRows(2 * b, 2).diagonal().setConstant(2);
            }
            return energy::BatchMatrix(2 * x);
        });
    Direct direct(input, output);
    Vec x(2), dx(2), gy(3);
    x << .1, .2;
    dx << .3, -.7;
    gy << .2, .4, -.9;
    const SpMat J = fallback.evaluate(x)->jacobian();
    EXPECT_TRUE(fallback.evaluate(x)->jvp(dx).isApprox(J * dx));
    EXPECT_TRUE(fallback.evaluate(x)->vjp(gy).isApprox(J.transpose() * gy));
    EXPECT_TRUE(direct.evaluate(x)->jvp(dx).isApprox(J * dx));
    EXPECT_TRUE(direct.evaluate(x)->vjp(gy).isApprox(J.transpose() * gy));
    EXPECT_NEAR(gy.dot(direct.evaluate(x)->jvp(dx)), dx.dot(direct.evaluate(x)->vjp(gy)), 1e-14);
    EXPECT_THROW(direct.evaluate(x)->jvp(Vec::Zero(3)), std::invalid_argument);
    direct.fail = true;
    EXPECT_THROW(direct.evaluate(x)->jvp(dx), std::domain_error);
    direct.fail = false;
    EXPECT_TRUE(direct.evaluate(x)->jvp(dx).isApprox(J * dx));
}
