#include "litepgo/fem/basis/3d/tet_p1_basis.hpp"
#include "litepgo/fem/kinematics/volumetric_element.hpp"
#include "litepgo/fem/local/local_response.hpp"
#include "litepgo/fem/quadrature/3d/tet_centroid_quadrature.hpp"
#include "litepgo/fem/response/interpolated_response.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <memory>

namespace litepgo::fem::test {
namespace {
template <int D> struct ValueOnly : LocalResponseT<D> {
    int num_inputs() const override {
        return 2;
    }
    int num_outputs() const override {
        return 3;
    }
    Eigen::VectorXd eval_value(const typename LocalResponseT<D>::Coordinate& xi) const override {
        return (Eigen::Vector3d() << 1 + xi[0], 2, 3).finished();
    }
};
struct Full : ValueOnly<3> {
    bool has_jacobian() const noexcept override {
        return true;
    }
    Eigen::MatrixXd eval_jacobian(const Eigen::Vector3d&) const override {
        Eigen::Matrix<double, 3, 2> J;
        J << 1, 2, 3, 4, 5, 6;
        return J;
    }
};
TEST(LocalResponse, DefaultCapabilitiesAndWeightedAddition) {
    ValueOnly<2> two;
    Eigen::VectorXd value = Eigen::VectorXd::Zero(3);
    two.accumulate(Eigen::Vector2d::Zero(), 2, value);
    EXPECT_EQ(value, Eigen::Vector3d(2, 4, 6));
    EXPECT_FALSE(two.has_jacobian());
    EXPECT_THROW(two.eval_jacobian(Eigen::Vector2d::Zero()), std::logic_error);
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(3, 2);
    EXPECT_THROW(two.accumulate(Eigen::Vector2d::Zero(), 1, value, &J), std::logic_error);
    Full full;
    const Eigen::Vector3d xi(.2, .3, .1);
    value.setOnes();
    J.setOnes();
    full.accumulate(xi, .5, value, &J);
    full.accumulate(xi, -.25, value, &J);
    EXPECT_TRUE(value.isApprox(Eigen::Vector3d::Ones() + .25 * full.eval_value(xi), 1e-14));
    EXPECT_TRUE(J.isApprox(Eigen::MatrixXd::Ones(3, 2) + .25 * full.eval_jacobian(xi), 1e-14));
}
TEST(LocalResponse, InvalidBuffersCoordinatesAndOverflow) {
    Full full;
    Eigen::VectorXd value = Eigen::VectorXd::Zero(3);
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(2, 3);
    EXPECT_THROW(full.accumulate(Eigen::Vector3d::Zero(), 1, value, &J), std::invalid_argument);
    EXPECT_EQ(value.norm(), 0);
    value.resize(2);
    EXPECT_THROW(full.accumulate(Eigen::Vector3d::Zero(), 1, value), std::invalid_argument);
    value.setZero(3);
    const double inf = std::numeric_limits<double>::infinity();
    EXPECT_THROW(full.accumulate(Eigen::Vector3d(inf, 0, 0), 1, value), std::invalid_argument);
    EXPECT_THROW(full.accumulate(Eigen::Vector3d::Zero(), inf, value), std::invalid_argument);
    EXPECT_THROW(full.accumulate(Eigen::Vector3d::Zero(), std::numeric_limits<double>::max(), value),
                 std::domain_error);
}
TEST(LocalResponse, RejectsMalformedLocalResultsBeforeAccumulating) {
    struct Bad : Full {
        bool bad_value = true;
        Eigen::VectorXd eval_value(const Eigen::Vector3d&) const override {
            return Eigen::VectorXd::Ones(bad_value ? 2 : 3);
        }
        Eigen::MatrixXd eval_jacobian(const Eigen::Vector3d&) const override {
            return Eigen::MatrixXd::Ones(3, 1);
        }
    } local;
    Eigen::VectorXd value = Eigen::VectorXd::Ones(3);
    Eigen::MatrixXd J = Eigen::MatrixXd::Ones(3, 2);
    EXPECT_THROW(local.accumulate(Eigen::Vector3d::Zero(), 1, value, &J), std::logic_error);
    local.bad_value = false;
    EXPECT_THROW(local.accumulate(Eigen::Vector3d::Zero(), 1, value, &J), std::logic_error);
    EXPECT_EQ(value, Eigen::VectorXd::Ones(3));
    EXPECT_EQ(J, Eigen::MatrixXd::Ones(3, 2));
}

TEST(LocalResponse, IntegratorUsesAccumulationAndChecksFinalOutputs) {
    Eigen::Matrix<double, 4, 3> vertices;
    vertices << 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1;
    auto X = std::make_shared<InterpolatedResponse<TetP1Basis>>(vertices);
    const VolumetricElement3D<TetCentroidQuadrature> element(*X);
    ValueOnly<3> no_jacobian;
    Eigen::MatrixXd J;
    EXPECT_THROW(element.eval(no_jacobian, &J), std::logic_error);
    struct Fast : Full {
        mutable int calls = 0;
        bool bad = false;
        void accumulate_into(const Eigen::Vector3d& xi, double w, StridedVecRef value,
                             StridedMatRef* jacobian) const override {
            ++calls;
            Full::accumulate_into(xi, w, value, jacobian);
            if (bad)
                value[0] = std::numeric_limits<double>::infinity();
        }
    } local;
    const auto value = element.eval(local, &J);
    EXPECT_EQ(local.calls, 1);
    EXPECT_TRUE(value.isApprox(local.eval_value(Eigen::Vector3d::Constant(.25)) / 6, 1e-14));
    EXPECT_TRUE(J.isApprox(local.eval_jacobian(Eigen::Vector3d::Zero()) / 6, 1e-14));
    local.bad = true;
    EXPECT_THROW(element.eval(local), std::domain_error);
}
} // namespace
} // namespace litepgo::fem::test

namespace litepgo::fem::test {
TEST(LocalResponse, OwningAndStridedBuffersUseCustomAccumulation) {
    struct Custom : Full {
        void accumulate_into(const Coordinate&, double weight, StridedVecRef value, StridedMatRef* J) const override {
            value.array() += 2 * weight;
            if (J)
                J->array() += 3 * weight;
        }
    } local;
    Eigen::VectorXd owned_value = Eigen::VectorXd::Constant(3, -5);
    Eigen::MatrixXd owned_J = Eigen::MatrixXd::Constant(3, 2, -5);
    local.accumulate(Eigen::Vector3d::Zero(), .5, owned_value, &owned_J);
    EXPECT_TRUE(owned_value.isConstant(-4));
    EXPECT_TRUE(owned_J.isConstant(-3.5));
    Eigen::MatrixXd values = Eigen::MatrixXd::Constant(2, 3, -5);
    Eigen::MatrixXd jacobians = Eigen::MatrixXd::Constant(5, 4, -5);
    StridedVecRef v(values.row(1).transpose());
    StridedMatRef J(jacobians.block(1, 1, 3, 2));
    local.accumulate_into(Eigen::Vector3d::Zero(), .5, v, &J);
    EXPECT_TRUE(v.isConstant(-4));
    EXPECT_TRUE(J.isConstant(-3.5));
    EXPECT_TRUE(values.row(0).isConstant(-5));
    EXPECT_TRUE(jacobians.row(0).isConstant(-5));
    EXPECT_TRUE(jacobians.col(3).isConstant(-5));
}
} // namespace litepgo::fem::test
