#include "../../testing/evaluate_energy.hpp"

#include "litepgo/dof/constant_dof_map.hpp"
#include "litepgo/dof/vertex_dof_map.hpp"
#include "litepgo/energy/response/affine_response.hpp"
#include "litepgo/fem/assembly/volumetric_fem_energy.hpp"
#include "litepgo/fem/assembly/volumetric_fem_force_fp_response.hpp"
#include "litepgo/fem/assembly/volumetric_fem_force_material_response.hpp"
#include "litepgo/fem/basis/3d/hex_q1_basis.hpp"
#include "litepgo/fem/basis/3d/tet_p1_basis.hpp"
#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/quadrature/3d/gauss_legendre_3d.hpp"
#include "litepgo/fem/quadrature/3d/tet_centroid_quadrature.hpp"
#include "litepgo/fem/response/interpolated_response.hpp"

#include "litepgo/fem/material/volumetric/pointwise_volumetric_adapter.hpp"
#include <atomic>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

namespace litepgo::fem::test {
namespace {

// NeoHookean evaluated through the batch protocol, recording each call.
struct Calls {
    std::atomic<int> eval = 0, steps = 0, points = 0;
};
class BatchNeoHookean : public VolumetricBatchMaterial {
  public:
    explicit BatchNeoHookean(std::shared_ptr<Calls> calls) : calls_(std::move(calls)) {}
    int num_parameters() const override {
        return NeoHookean::num_parameters;
    }
    void eval(const VolumetricMaterialBatch& in, VolumetricMaterialBatchOutputs& out) const override {
        ++calls_->eval;
        calls_->points += static_cast<int>(in.size());
        const Eigen::Index n = in.size();
        if (out.W)
            out.W->resize(n);
        if (out.dW_dF)
            out.dW_dF->resize(9, n);
        if (out.d2W_dF2)
            out.d2W_dF2->resize(81, n);
        if (out.d2W_dF_dtheta)
            out.d2W_dF_dtheta->resize(9 * num_parameters(), n);
        if (out.d2W_dF_dFp)
            out.d2W_dF_dFp->resize(81, n);
        for (Eigen::Index p = 0; p < n; ++p) {
            const VolumetricMaterialInputs input = point(in, p);
            if (out.W)
                (*out.W)[p] = NeoHookean::eval_W(input);
            if (out.dW_dF)
                out.dW_dF->col(p) = NeoHookean::eval_dW_dF(input).reshaped();
            if (out.d2W_dF2)
                out.d2W_dF2->col(p) = NeoHookean::eval_d2W_dF2(input).reshaped();
            if (out.d2W_dF_dtheta)
                out.d2W_dF_dtheta->col(p) = NeoHookean::eval_d2W_dF_dtheta(input).reshaped();
            if (out.d2W_dF_dFp)
                out.d2W_dF_dFp->col(p) = NeoHookean::eval_d2W_dF_dFp(input).reshaped();
        }
    }
    Vec eval_feasible_step(const VolumetricMaterialBatch& in,
                           const Eigen::Matrix<double, 9, Eigen::Dynamic>& dF) const override {
        ++calls_->steps;
        Vec steps(in.size());
        for (Eigen::Index p = 0; p < in.size(); ++p) {
            const VolumetricMaterialInputs input = point(in, p);
            const VolumetricMaterialInputs direction{dF.col(p).reshaped(3, 3), Eigen::Matrix3d::Zero(),
                                                     Vec::Zero(input.theta.size())};
            steps[p] = NeoHookean::eval_feasible_step(input, direction);
        }
        return steps;
    }

  private:
    static VolumetricMaterialInputs point(const VolumetricMaterialBatch& in, Eigen::Index p) {
        return {in.F.col(p).reshaped(3, 3), in.Fp.col(p).reshaped(3, 3), in.theta.col(p)};
    }
    std::shared_ptr<Calls> calls_;
};
static_assert(std::is_base_of_v<VolumetricBatchMaterial, BatchNeoHookean>);
static_assert(!std::is_base_of_v<VolumetricBatchMaterial, NeoHookean>);

using Responses = std::vector<std::shared_ptr<litepgo::energy::Response>>;
std::shared_ptr<litepgo::energy::Response> constant(const Vec& value) {
    return std::make_shared<litepgo::energy::AffineResponse>(litepgo::SpMat(value.size(), 3), value);
}

template <class Basis, class Q> struct Fixture {
    mesh::TetTopology tets{5, {{0, 1, 2, 3}, {1, 2, 3, 4}}};
    mesh::HexTopology hexes{12, {{0, 1, 2, 3, 4, 5, 6, 7}, {1, 8, 9, 2, 5, 10, 11, 6}}};
    static constexpr bool is_tet = std::same_as<Basis, TetP1Basis>;
    dof::VertexDofMap map = is_tet ? dof::VertexDofMap(tets, 3) : dof::VertexDofMap(hexes, 3);
    Responses X, Fp, theta;
    Vec u;
    const auto& topology() const {
        if constexpr (is_tet)
            return tets;
        else
            return hexes;
    }
    Fixture() {
        Eigen::MatrixXd nodes;
        if constexpr (is_tet) {
            nodes.resize(5, 3);
            nodes << 0, 0, 0, 1.2, 0, 0, 0, 1, 0, 0, 0, .8, 1.2, 1, .8;
        } else {
            nodes.resize(12, 3);
            nodes << 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 3, 0, 0, 3, 1, 0, 3, 0, 1,
                3, 1, 1;
        }
        const Vec positions = nodes.reshaped<Eigen::RowMajor>();
        u = Vec(positions.size());
        for (Eigen::Index i = 0; i < u.size(); ++i)
            u[i] = 0.04 * std::sin(1.3 * i + 0.2);
        for (int e = 0; e < map.num_elements(); ++e) {
            X.push_back(std::make_shared<InterpolatedResponse<Basis>>(map.gather(e, positions)));
            Eigen::Matrix3d plastic = Eigen::Matrix3d::Identity();
            plastic(0, 0) = 1.1 + .05 * e;
            plastic(1, 2) = .1;
            Fp.push_back(constant(plastic.reshaped()));
            theta.push_back(constant(Eigen::Vector2d(2 + e, 3)));
        }
    }
};

template <class Basis, class Q> void check_energy() {
    Fixture<Basis, Q> f;
    auto calls = std::make_shared<Calls>();
    const VolumetricFemEnergy<Basis, Q> native(std::make_shared<PointwiseVolumetricAdapter<NeoHookean>>(), f.map, f.X,
                                               f.Fp, f.theta);
    for (int batch_size : {1, 256}) {
        SCOPED_TRACE(batch_size);
        const VolumetricFemEnergy<Basis, Q> batched(std::make_shared<BatchNeoHookean>(calls), f.map, f.X, f.Fp, f.theta,
                                                    batch_size);
        for (auto kind : {litepgo::energy::HessianKind::Exact, litepgo::energy::HessianKind::Projected}) {
            Vec g0, g1;
            SpMat H0, H1;
            calls->eval = 0;
            const double E0 = litepgo::testing::evaluate_energy(native, f.u, &g0, &H0, kind);
            const double E1 = litepgo::testing::evaluate_energy(batched, f.u, &g1, &H1, kind);
            EXPECT_EQ(calls->eval, batch_size == 1 ? f.map.num_elements() : 1);
            EXPECT_NEAR(E0, E1, 1e-13 * (1 + std::abs(E0)));
            EXPECT_LT((g0 - g1).norm(), 1e-12 * (1 + g0.norm()));
            EXPECT_LT(Eigen::MatrixXd(H0 - H1).norm(), 1e-12 * (1 + Eigen::MatrixXd(H0).norm()));
        }
        const Vec dx = 5 * f.u;
        EXPECT_DOUBLE_EQ(native.max_feasible_step(f.u, dx), batched.max_feasible_step(f.u, dx));
        EXPECT_GT(calls->steps, 0);
        EXPECT_NEAR(native.eval_value(f.u), batched.eval_value(f.u), 1e-12);
    }
}

TEST(VolumetricFemBatchMaterial, EnergyMatchesPointwiseMaterial) {
    check_energy<TetP1Basis, TetCentroidQuadrature>();
    check_energy<HexQ1Basis, GaussLegendre3D<2>>();
}

template <class Basis, class Q> void check_forces() {
    Fixture<Basis, Q> f;
    auto calls = std::make_shared<Calls>();
    const int count = f.map.num_elements();
    const dof::ConstantDofMap material_map(f.topology(), 2), plastic_map(f.topology(), 9);
    VolumetricFemForceMaterialResponse<Basis, Q> native_theta(
        std::make_shared<PointwiseVolumetricAdapter<NeoHookean>>(), material_map, f.map, f.X, f.u, f.Fp);
    VolumetricFemForceMaterialResponse<Basis, Q> batch_theta(std::make_shared<BatchNeoHookean>(calls), material_map,
                                                             f.map, f.X, f.u, f.Fp, 256);
    const Vec theta = Eigen::Vector2d(2.5, 1.5);
    calls->eval = 0;
    EXPECT_LT((native_theta.eval_value(theta) - batch_theta.eval_value(theta)).norm(), 1e-12);
    EXPECT_LT(Eigen::MatrixXd(native_theta.eval_jacobian(theta) - batch_theta.eval_jacobian(theta)).norm(), 1e-12);
    EXPECT_EQ(calls->eval, 2);

    VolumetricFemForceFpResponse<Basis, Q> native_fp(std::make_shared<PointwiseVolumetricAdapter<NeoHookean>>(),
                                                     plastic_map, f.map, f.X, f.u, f.theta);
    VolumetricFemForceFpResponse<Basis, Q> batch_fp(std::make_shared<BatchNeoHookean>(calls), plastic_map, f.map, f.X,
                                                    f.u, f.theta, 1);
    Eigen::Matrix3d plastic = Eigen::Matrix3d::Identity();
    plastic(0, 1) = 0.07;
    plastic(2, 2) = 0.9;
    const Vec packed = plastic.reshaped();
    calls->eval = 0;
    EXPECT_LT((native_fp.eval_value(packed) - batch_fp.eval_value(packed)).norm(), 1e-12);
    EXPECT_LT(Eigen::MatrixXd(native_fp.eval_jacobian(packed) - batch_fp.eval_jacobian(packed)).norm(), 1e-12);
    EXPECT_EQ(calls->eval, 2 * count);
}

TEST(VolumetricFemBatchMaterial, ForceResponsesMatchPointwiseMaterial) {
    check_forces<TetP1Basis, TetCentroidQuadrature>();
    check_forces<HexQ1Basis, GaussLegendre3D<2>>();
}

// Requests only what each operation needs.
using Request = std::array<bool, 5>;
class RecordingMaterial : public BatchNeoHookean {
  public:
    RecordingMaterial(std::shared_ptr<Calls> calls, std::shared_ptr<std::vector<Request>> requests)
        : BatchNeoHookean(std::move(calls)), requests_(std::move(requests)) {}
    void eval(const VolumetricMaterialBatch& in, VolumetricMaterialBatchOutputs& out) const override {
        requests_->push_back({out.W != nullptr, out.dW_dF != nullptr, out.d2W_dF2 != nullptr,
                              out.d2W_dF_dtheta != nullptr, out.d2W_dF_dFp != nullptr});
        BatchNeoHookean::eval(in, out);
    }

  private:
    std::shared_ptr<std::vector<Request>> requests_;
};

TEST(VolumetricFemBatchMaterial, RequestsOnlyNeededOutputs) {
    Fixture<TetP1Basis, TetCentroidQuadrature> f;
    auto calls = std::make_shared<Calls>();
    auto requests = std::make_shared<std::vector<Request>>();
    const auto material = std::make_shared<RecordingMaterial>(calls, requests);
    const VolumetricFemEnergy<TetP1Basis, TetCentroidQuadrature> energy(material, f.map, f.X, f.Fp, f.theta, 256);
    energy.eval_value(f.u);
    energy.eval_gradient(f.u);
    Vec g;
    SpMat H;
    litepgo::testing::evaluate_energy(energy, f.u, &g, &H);
    const dof::ConstantDofMap material_map(f.topology(), 2), plastic_map(f.topology(), 9);
    VolumetricFemForceMaterialResponse<TetP1Basis, TetCentroidQuadrature> theta_force(material, material_map, f.map,
                                                                                      f.X, f.u, f.Fp, 256);
    VolumetricFemForceFpResponse<TetP1Basis, TetCentroidQuadrature> fp_force(material, plastic_map, f.map, f.X, f.u,
                                                                             f.theta, 256);
    theta_force.eval_value(Eigen::Vector2d(2, 3));
    theta_force.eval_jacobian(Eigen::Vector2d(2, 3));
    fp_force.eval_jacobian(Eigen::Matrix3d::Identity().reshaped());
    const std::vector<Request> expected{{true, false, false, false, false}, {true, true, false, false, false},
                                        {true, true, true, false, false},   {false, true, false, false, false},
                                        {false, true, false, true, false},  {false, true, false, false, true}};
    EXPECT_EQ(*requests, expected);
}

TEST(VolumetricFemBatchMaterial, RejectsInvalidMaterialOutputs) {
    struct Bad : BatchNeoHookean {
        using BatchNeoHookean::BatchNeoHookean;
        void eval(const VolumetricMaterialBatch& in, VolumetricMaterialBatchOutputs& out) const override {
            BatchNeoHookean::eval(in, out);
            if (out.W)
                (*out.W)[0] = std::numeric_limits<double>::quiet_NaN();
        }
    };
    Fixture<TetP1Basis, TetCentroidQuadrature> f;
    const VolumetricFemEnergy<TetP1Basis, TetCentroidQuadrature> energy(
        std::make_shared<Bad>(std::make_shared<Calls>()), f.map, f.X, f.Fp, f.theta, 256);
    EXPECT_THROW(energy.eval_value(f.u), std::domain_error);
}

} // namespace
} // namespace litepgo::fem::test
