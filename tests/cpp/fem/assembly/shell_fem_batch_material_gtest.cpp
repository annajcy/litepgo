#include "../../testing/evaluate_energy.hpp"

#include "litepgo/dof/constant_dof_map.hpp"
#include "litepgo/fem/assembly/shell_fem_energy.hpp"
#include "litepgo/fem/assembly/shell_fem_force_fp_response.hpp"
#include "litepgo/fem/assembly/shell_fem_force_material_response.hpp"
#include "litepgo/fem/material/shell/koiter_stvk.hpp"
#include "litepgo/mesh/triangle_mid_edge_stencils.hpp"

#include "litepgo/fem/material/shell/pointwise_shell_adapter.hpp"
#include <atomic>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <vector>

namespace litepgo::fem::test {
namespace {

// KoiterStVK evaluated through the batch protocol, counting calls.
class BatchKoiter : public ShellBatchMaterial {
  public:
    explicit BatchKoiter(std::shared_ptr<std::atomic<int>> calls, bool poison = false)
        : calls_(std::move(calls)), poison_(poison) {}
    int num_parameters() const override {
        return KoiterStVK::num_parameters;
    }
    void eval(const ShellMaterialBatch& in, ShellMaterialBatchOutputs& out) const override {
        ++*calls_;
        const Eigen::Index n = in.size();
        if (out.W)
            out.W->resize(n);
        if (out.dW_dab)
            out.dW_dab->resize(6, n);
        if (out.d2W_dab2)
            out.d2W_dab2->resize(36, n);
        if (out.d2W_dab_dtheta)
            out.d2W_dab_dtheta->resize(6 * num_parameters(), n);
        if (out.d2W_dab_dabarbbar)
            out.d2W_dab_dabarbbar->resize(36, n);
        for (Eigen::Index p = 0; p < n; ++p) {
            const ShellMaterialInputs input{unpack(in.ab.col(p).head<3>()),
                                            unpack(in.ab.col(p).tail<3>()),
                                            unpack(in.ab_bar.col(p).head<3>()),
                                            unpack(in.ab_bar.col(p).tail<3>()),
                                            in.thickness[p],
                                            in.theta.col(p)};
            if (out.W)
                (*out.W)[p] = poison_ ? std::numeric_limits<double>::infinity() : KoiterStVK::eval_W(input);
            if (out.dW_dab)
                out.dW_dab->col(p) = KoiterStVK::eval_dW_dab(input);
            if (out.d2W_dab2)
                out.d2W_dab2->col(p) = KoiterStVK::eval_d2W_dab2(input).reshaped();
            if (out.d2W_dab_dtheta)
                out.d2W_dab_dtheta->col(p) = KoiterStVK::eval_d2W_dab_dtheta(input).reshaped();
            if (out.d2W_dab_dabarbbar)
                out.d2W_dab_dabarbbar->col(p) = KoiterStVK::eval_d2W_dab_dabarbbar(input).reshaped();
        }
    }

  private:
    static Eigen::Matrix2d unpack(const Eigen::Vector3d& packed) {
        return (Eigen::Matrix2d() << packed[0], packed[1], packed[1], packed[2]).finished();
    }
    std::shared_ptr<std::atomic<int>> calls_;
    bool poison_;
};
static_assert(std::is_base_of_v<ShellBatchMaterial, BatchKoiter>);
static_assert(!std::is_base_of_v<ShellBatchMaterial, KoiterStVK>);

struct Patch {
    mesh::TriangleTopology topology{6, {{0, 1, 2}, {2, 1, 3}, {0, 2, 4}, {1, 0, 5}}};
    dof::StencilDofMap map{6, mesh::make_triangle_mid_edge_stencils(topology).vertices, 3};
    std::vector<TriangleMidEdgeRestData> rest;
    std::vector<double> thickness{.1, .2, .15, .3};
    std::vector<Vec> theta{Eigen::Vector2d(2, 3), Eigen::Vector2d(4, 5), Eigen::Vector2d(-.5, 2),
                           Eigen::Vector2d(8, 9)};
    std::vector<Eigen::Matrix3d> Fp;
    Vec u = .03 * Vec::LinSpaced(18, -1, 1);
    Patch() {
        Eigen::Matrix<double, 6, 3> X;
        X << 0, 0, 0, 2, 0, 0, 0, 1, 0, 2, 2, .1, -1, 0, -.1, 0, -2, .2;
        const Vec x = X.reshaped<Eigen::RowMajor>();
        const auto stencil = mesh::make_triangle_mid_edge_stencils(topology);
        for (int e = 0; e < 4; ++e) {
            rest.emplace_back(map.gather(e, x), stencil.opposite[e]);
            Eigen::Matrix3d P = Eigen::Matrix3d::Identity();
            P(0, 0) = 1.05 + .02 * e;
            P(1, 0) = .03;
            Fp.push_back(P);
        }
        u[11] += .2;
        u[14] -= .15;
    }
};

TEST(ShellFemBatchMaterial, EnergyMatchesPointwiseMaterial) {
    Patch p;
    auto calls = std::make_shared<std::atomic<int>>(0);
    const ShellFemEnergy native(std::make_shared<PointwiseShellAdapter<KoiterStVK>>(), p.map, p.rest, p.thickness,
                                p.theta, p.Fp);
    for (int batch_size : {1, 256}) {
        SCOPED_TRACE(batch_size);
        const ShellFemEnergy batched(std::make_shared<BatchKoiter>(calls), p.map, p.rest, p.thickness, p.theta, p.Fp,
                                     batch_size);
        for (auto kind : {litepgo::energy::HessianKind::Exact, litepgo::energy::HessianKind::Projected}) {
            Vec g0, g1;
            SpMat H0, H1;
            *calls = 0;
            const double E0 = litepgo::testing::evaluate_energy(native, p.u, &g0, &H0, kind);
            const double E1 = litepgo::testing::evaluate_energy(batched, p.u, &g1, &H1, kind);
            // Batches group faces by stencil size; this patch has two sizes.
            EXPECT_EQ(*calls, batch_size == 1 ? 4 : 2);
            EXPECT_NEAR(E0, E1, 1e-13 * (1 + std::abs(E0)));
            EXPECT_LT((g0 - g1).norm(), 1e-12 * (1 + g0.norm()));
            EXPECT_LT(Eigen::MatrixXd(H0 - H1).norm(), 1e-12 * (1 + Eigen::MatrixXd(H0).norm()));
        }
        const Vec dx = 3 * p.u;
        EXPECT_DOUBLE_EQ(native.max_feasible_step(p.u, dx), batched.max_feasible_step(p.u, dx));
    }
    EXPECT_THROW(
        ShellFemEnergy(std::make_shared<BatchKoiter>(calls, true), p.map, p.rest, p.thickness, p.theta, p.Fp, 256),
        std::domain_error);
}

TEST(ShellFemBatchMaterial, ForceResponsesMatchPointwiseMaterial) {
    Patch p;
    auto calls = std::make_shared<std::atomic<int>>(0);
    const Vec theta = Eigen::Vector2d(2.5, 1.5);
    ShellFemForceMaterialResponse native_theta(std::make_shared<PointwiseShellAdapter<KoiterStVK>>(),
                                               dof::ConstantDofMap(p.topology, 2), p.map, p.rest, p.u, p.thickness);
    ShellFemForceMaterialResponse batch_theta(std::make_shared<BatchKoiter>(calls), dof::ConstantDofMap(p.topology, 2),
                                              p.map, p.rest, p.u, p.thickness, 256);
    EXPECT_LT((native_theta.evaluate(theta)->value() - batch_theta.evaluate(theta)->value()).norm(), 1e-12);
    EXPECT_LT(
        Eigen::MatrixXd(native_theta.evaluate(theta)->jacobian() - batch_theta.evaluate(theta)->jacobian()).norm(),
        1e-12);
    EXPECT_EQ(*calls, 4); // Two stencil-size batches per evaluation.

    Eigen::Matrix3d P;
    P << 1.12, .07, -.02, .03, .94, .06, .01, -.03, 1.08;
    const Vec packed = P.reshaped();
    ShellFemForceFpResponse native_fp(std::make_shared<PointwiseShellAdapter<KoiterStVK>>(),
                                      dof::ConstantDofMap(p.topology, 9), p.map, p.rest, p.u, p.thickness, p.theta);
    ShellFemForceFpResponse batch_fp(std::make_shared<BatchKoiter>(calls), dof::ConstantDofMap(p.topology, 9), p.map,
                                     p.rest, p.u, p.thickness, p.theta, 256);
    EXPECT_LT((native_fp.evaluate(packed)->value() - batch_fp.evaluate(packed)->value()).norm(), 1e-12);
    EXPECT_LT(Eigen::MatrixXd(native_fp.evaluate(packed)->jacobian() - batch_fp.evaluate(packed)->jacobian()).norm(),
              1e-12);
    EXPECT_EQ(*calls, 8);
}

} // namespace
} // namespace litepgo::fem::test
