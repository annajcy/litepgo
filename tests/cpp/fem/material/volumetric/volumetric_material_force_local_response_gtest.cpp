#include "litepgo/fem/basis/3d/tet_p1_basis.hpp"
#include "litepgo/fem/field/interpolated_field.hpp"
#include "litepgo/fem/kinematics/volumetric_element.hpp"
#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/material/volumetric/volumetric_fp_force_local_response.hpp"
#include "litepgo/fem/material/volumetric/volumetric_material_force_local_response.hpp"
#include "litepgo/fem/material/volumetric/volumetric_material_local_energy.hpp"
#include "litepgo/fem/quadrature/3d/tet_centroid_quadrature.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>

namespace litepgo::fem::test {
namespace {
using Local = VolumetricMaterialForceLocalResponse3D<NeoHookean>;
struct Field : Field3D {
    Eigen::VectorXd data;
    Eigen::MatrixXd D;
    mutable int derivative_calls = 0;
    bool reject_derivative = false;
    Field(Eigen::VectorXd value, int dofs) : data(std::move(value)), D(Eigen::MatrixXd::Zero(data.size(), dofs)) {}
    int num_components() const override {
        return static_cast<int>(data.size());
    }
    int num_dofs() const override {
        return static_cast<int>(D.cols());
    }
    Eigen::VectorXd eval_value(const Eigen::Vector3d&) const override {
        return data;
    }
    Eigen::MatrixXd eval_dvalue_dcoeffs(const Eigen::Vector3d&) const override {
        ++derivative_calls;
        if (reject_derivative)
            throw std::logic_error("unexpected field derivative");
        return D;
    }
};
struct Fixture {
    Field F{Eigen::VectorXd::Zero(9), 4}, Fp{Eigen::VectorXd::Zero(9), 0}, theta{Eigen::Vector2d(2.1, 3.05), 3};
    Fixture() {
        Eigen::Matrix3d f, fp;
        f << 1.2, .3, -.1, .1, .9, .2, -.05, .1, 1.1;
        fp << 1.3, .2, .1, 0, .9, -.1, .1, .2, 1.2;
        F.data = f.reshaped();
        Fp.data = fp.reshaped();
        for (int r = 0; r < 9; ++r)
            for (int c = 0; c < 4; ++c)
                F.D(r, c) = .1 * (r + 1) * (c + 1) + (r == c ? 1 : 0);
        theta.D << 1, 0, .2, 0, 1, .1;
        Fp.reject_derivative = true;
    }
};
TEST(MaterialLocalResponse, PullbackMatchesEnergyAndExactParameterLinearity) {
    static_assert(!std::is_constructible_v<Local, Field&&, const Field&, const Field&>);
    static_assert(!std::is_constructible_v<Local, const Field&, Field&&, const Field&>);
    static_assert(!std::is_constructible_v<Local, const Field&, const Field&, Field&&>);
    Fixture f;
    const Local local(f.F, f.Fp, f.theta);
    const VolumetricMaterialLocalEnergyF3D<NeoHookean> energy(f.F, f.Fp, f.theta);
    const Eigen::Vector3d xi(.2, .1, .3);
    EXPECT_EQ(local.num_inputs(), 3);
    EXPECT_EQ(local.num_outputs(), 4);
    EXPECT_TRUE(local.has_jacobian());
    const auto value = local.eval_value(xi);
    const auto J = local.eval_jacobian(xi);
    EXPECT_TRUE(value.isApprox(energy.eval_gradient(xi), 1e-13));
    EXPECT_TRUE((J * Eigen::Vector3d(2, 3, .5)).isApprox(value, 1e-13));
    EXPECT_TRUE(J.col(2).isApprox(.2 * J.col(0) + .1 * J.col(1), 1e-13));
    const Eigen::Vector3d coeffs(1, 4, .25);
    f.theta.data = f.theta.D * coeffs;
    EXPECT_TRUE(local.eval_value(xi).isApprox(J * coeffs, 1e-13));
    EXPECT_EQ((local.eval_jacobian(xi) - J).norm(), 0);
    EXPECT_EQ(f.Fp.derivative_calls, 0);
    f.theta.reject_derivative = true;
    EXPECT_NO_THROW(local.eval_value(xi));
    EXPECT_THROW(local.eval_jacobian(xi), std::logic_error);
}
TEST(MaterialLocalResponse, AccumulationAndIntegrationMatchEnergyWithVolumeAppliedOnce) {
    Fixture f;
    const Local local(f.F, f.Fp, f.theta);
    const Eigen::Vector3d xi = Eigen::Vector3d::Constant(.25);
    Eigen::VectorXd value = Eigen::VectorXd::Ones(4), fast = value;
    Eigen::MatrixXd J = Eigen::MatrixXd::Ones(4, 3), fast_J = J;
    local.accumulate(xi, .4, value, &J);
    StridedVecRef value_view(fast);
    StridedMatRef J_view(fast_J);
    local.accumulate_into(xi, .4, value_view, &J_view);
    EXPECT_EQ((value - fast).norm(), 0);
    EXPECT_EQ((J - fast_J).norm(), 0);
    EXPECT_TRUE(value.isApprox(Eigen::VectorXd::Ones(4) + .4 * local.eval_value(xi), 1e-13));
    EXPECT_TRUE(J.isApprox(Eigen::MatrixXd::Ones(4, 3) + .4 * local.eval_jacobian(xi), 1e-13));
    Eigen::Matrix<double, 4, 3> vertices;
    vertices << 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2;
    const InterpolatedField3D<TetP1Basis> X(vertices);
    const VolumetricElement3D<TetCentroidQuadrature> element(X);
    const auto integrated = element.eval(local, &J);
    Eigen::VectorXd gradient;
    element.eval(VolumetricMaterialLocalEnergyF3D<NeoHookean>(f.F, f.Fp, f.theta), &gradient);
    EXPECT_TRUE(integrated.isApprox(gradient, 1e-13));
    EXPECT_TRUE(integrated.isApprox((8. / 6) * local.eval_value(xi), 1e-13));
    EXPECT_TRUE(J.isApprox((8. / 6) * local.eval_jacobian(xi), 1e-13));
    // Direct evaluation and integration reject invalid field derivatives consistently.
    f.F.D(0, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(local.eval_value(xi), std::invalid_argument);
    EXPECT_THROW(element.eval(local, &J), std::invalid_argument);
    f.F.D(0, 0) = 1;
    f.theta.D(0, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_NO_THROW(element.eval(local));
    EXPECT_THROW(local.eval_jacobian(xi), std::invalid_argument);
    EXPECT_THROW(element.eval(local, &J), std::invalid_argument);
}
TEST(MaterialLocalResponse, RejectsInvalidFieldsBuffersAndNumericalResults) {
    Fixture f;
    Field wrong(Eigen::VectorXd::Ones(8), 2);
    EXPECT_THROW((Local(wrong, f.Fp, f.theta)), std::invalid_argument);
    EXPECT_THROW((Local(f.F, wrong, f.theta)), std::invalid_argument);
    EXPECT_THROW((Local(f.F, f.Fp, wrong)), std::invalid_argument);
    const Local local(f.F, f.Fp, f.theta);
    Eigen::VectorXd value = Eigen::VectorXd::Zero(4);
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(3, 4);
    const Eigen::Vector3d xi = Eigen::Vector3d::Zero();
    EXPECT_THROW(local.accumulate(xi, 1, value, &J), std::invalid_argument);
    EXPECT_EQ(value.norm(), 0);
    EXPECT_THROW(local.accumulate(xi, std::numeric_limits<double>::infinity(), value), std::invalid_argument);
    f.theta.data[1] = 0;
    EXPECT_THROW(local.eval_value(xi), std::invalid_argument);
    f.theta.data[1] = 3;
    f.theta.D.resize(1, 3);
    EXPECT_THROW(local.eval_jacobian(xi), std::invalid_argument);
    f.theta.D.setOnes(2, 3);
    f.F.D.setConstant(std::numeric_limits<double>::max());
    EXPECT_THROW(local.eval_value(xi), std::domain_error);
    EXPECT_THROW(local.eval_jacobian(xi), std::domain_error);
}
} // namespace
} // namespace litepgo::fem::test

namespace litepgo::fem::test {
TEST(VolumetricFpForceLocalResponse, RectangularPullbackViewsAndInactiveParameters) {
    using Plastic = VolumetricFpForceLocalResponse3D<NeoHookean>;
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    Field F((1.1 * I).reshaped(), 4), Fp(I.reshaped(), 3), theta(Eigen::Vector2d(2, 3), 0);
    F.D.setRandom();
    Fp.D.setRandom();
    theta.reject_derivative = true;
    const Plastic local(F, Fp, theta);
    static_assert(!std::is_constructible_v<Plastic, Field&&, const Field&, const Field&>);
    static_assert(!std::is_constructible_v<Plastic, const Field&, Field&&, const Field&>);
    static_assert(!std::is_constructible_v<Plastic, const Field&, const Field&, Field&&>);
    const Eigen::Vector3d xi(.2, .3, .1);
    const VolumetricMaterialInputs inputs{1.1 * I, I, Eigen::Vector2d(2, 3)};
    EXPECT_TRUE(local.eval_value(xi).isApprox(F.D.transpose() * NeoHookean::eval_dW_dF(inputs).reshaped(), 1e-13));
    const Eigen::MatrixXd expected = F.D.transpose() * NeoHookean::eval_d2W_dF_dFp(inputs) * Fp.D;
    EXPECT_TRUE(local.eval_jacobian(xi).isApprox(expected, 1e-13));
    for (int c = 0; c < 3; ++c) {
        Fp.data = I.reshaped() + 1e-6 * Fp.D.col(c);
        const auto plus = local.eval_value(xi);
        Fp.data = I.reshaped() - 1e-6 * Fp.D.col(c);
        const auto minus = local.eval_value(xi);
        EXPECT_TRUE(expected.col(c).isApprox((plus - minus) / 2e-6, 1e-8));
    }
    Fp.data = I.reshaped();
    Eigen::MatrixXd storage = Eigen::MatrixXd::Constant(6, 5, 7);
    StridedVecRef value(storage.col(0).segment(1, 4));
    StridedMatRef J(storage.block(1, 1, 4, 3));
    value.setZero();
    J.setZero();
    local.accumulate_into(xi, .3, value, &J);
    EXPECT_TRUE(value.isApprox(.3 * local.eval_value(xi), 1e-13));
    EXPECT_TRUE(J.isApprox(.3 * expected, 1e-13));
    EXPECT_TRUE(storage.row(0).isConstant(7));
    EXPECT_TRUE(storage.row(5).isConstant(7));
    EXPECT_TRUE(storage.col(4).isConstant(7));
    Fp.reject_derivative = true;
    EXPECT_NO_THROW(local.eval_value(xi));
    EXPECT_THROW(local.eval_jacobian(xi), std::logic_error);
    Fp.reject_derivative = false;
    Fp.D.resize(8, 3);
    EXPECT_THROW(local.eval_jacobian(xi), std::invalid_argument);
    Fp.D.resize(9, 0);
    EXPECT_THROW((Plastic(F, Fp, theta)), std::invalid_argument);
}
} // namespace litepgo::fem::test
