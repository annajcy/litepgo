#include "litepgo/fem/basis/1d/cubic_hermite_basis.hpp"
#include "litepgo/fem/basis/1d/linear_lagrange_basis.hpp"
#include "litepgo/fem/basis/2d/quad_hermite_basis.hpp"
#include "litepgo/fem/basis/2d/triangle_p1_basis.hpp"
#include "litepgo/fem/basis/3d/hex_hermite_basis.hpp"
#include "litepgo/fem/basis/3d/hex_q1_basis.hpp"
#include "litepgo/fem/basis/3d/tet_p1_basis.hpp"
#include "litepgo/fem/basis/basis.hpp"
#include "litepgo/fem/response/interpolated_response.hpp"
#include "litepgo/fem/response/sampling_response.hpp"

#include <gtest/gtest.h>

#include <array>

namespace litepgo::fem::test {

namespace {

struct ConstantBasis {
    static constexpr int num_basis_functions = 1;
    static CoefficientInfo<1> coefficient_info(int i) {
        if (i != 0)
            throw std::out_of_range("ConstantBasis coefficient index");
        return {0.0, {0}};
    }
    static Eigen::Matrix<double, 1, 1> eval_N(double) {
        return Eigen::Matrix<double, 1, 1>(1.0);
    }
    static Eigen::Matrix<double, 1, 1> eval_dN_dxi(double) {
        return Eigen::Matrix<double, 1, 1>(0.0);
    }
    static Eigen::Matrix<double, 1, 1> eval_d2N_dxi2(double) {
        return Eigen::Matrix<double, 1, 1>(0.0);
    }
};

struct InstanceOnly1D : ConstantBasis {
    Eigen::Matrix<double, 1, 1> eval_N(double) const;
};
struct InstanceOnly3D : HexQ1Basis {
    Eigen::Matrix<double, 8, 1> eval_N(const Eigen::Vector3d&) const;
};
struct MissingBasis {};
struct MissingCoefficientInfo : ConstantBasis {
    static CoefficientInfo<1> coefficient_info(int) = delete;
};
struct WrongCoefficientInfo : ConstantBasis {
    static double coefficient_info(int);
};
struct MissingEvaluate : ConstantBasis {
    static Eigen::Matrix<double, 1, 1> eval_N(double) = delete;
};
struct NonConstantCount : ConstantBasis {
    inline static int num_basis_functions = 1;
};
struct ZeroCount : ConstantBasis {
    static constexpr int num_basis_functions = 0;
};
struct NegativeCount : ConstantBasis {
    static constexpr int num_basis_functions = -1;
};
struct WrongReturn : ConstantBasis {
    static double eval_N(double);
};
struct WrongGradientSize : ConstantBasis {
    static Eigen::Matrix<double, 2, 1> eval_dN_dxi(double);
};
struct WrongValueScalar : ConstantBasis {
    static Eigen::Matrix<float, 1, 1> eval_N(double);
};
struct Wrong1DGradientColumns : LinearLagrangeBasis {
    static Eigen::Matrix<double, 2, 2> eval_dN_dxi(double);
};
struct WrongGradientColumns : HexQ1Basis {
    static Eigen::Matrix<double, 8, 2> eval_dN_dxi(const Eigen::Vector3d&);
};
struct WrongGradientRows : HexQ1Basis {
    static Eigen::Matrix<double, 7, 3> eval_dN_dxi(const Eigen::Vector3d&);
};
struct Wrong3DReturn : HexQ1Basis {
    static double eval_N(const Eigen::Vector3d&);
};
struct MissingHessian : ConstantBasis {
    static Eigen::Matrix<double, 1, 1> eval_d2N_dxi2(double) = delete;
};
struct WrongHessianScalar : ConstantBasis {
    static Eigen::Matrix<float, 1, 1> eval_d2N_dxi2(double);
};
struct WrongHessianColumns : HexQ1Basis {
    static Eigen::Matrix<double, 8, 5> eval_d2N_dxi2(const Eigen::Vector3d&);
};
struct WrongHessianRows : HexQ1Basis {
    static Eigen::Matrix<double, 7, 6> eval_d2N_dxi2(const Eigen::Vector3d&);
};

static_assert(Basis1D<LinearLagrangeBasis>);
static_assert(Basis1D<CubicHermiteBasis>);
static_assert(Basis1D<ConstantBasis>);
static_assert(Basis2D<QuadHermiteBasis>);
static_assert(Basis3D<HexQ1Basis>);
static_assert(Basis3D<HexHermiteBasis>);
static_assert(Basis3D<TetP1Basis>);
static_assert(Basis<HexQ1Basis, 3>);
static_assert(!Basis1D<InstanceOnly1D>);
static_assert(!Basis3D<InstanceOnly3D>);
static_assert(!Basis1D<MissingBasis>);
static_assert(!Basis1D<MissingCoefficientInfo>);
static_assert(!Basis1D<WrongCoefficientInfo>);
static_assert(!Basis1D<MissingEvaluate>);
static_assert(!Basis1D<NonConstantCount>);
static_assert(!Basis1D<ZeroCount>);
static_assert(!Basis1D<NegativeCount>);
static_assert(!Basis1D<WrongReturn>);
static_assert(!Basis1D<WrongGradientSize>);
static_assert(!Basis1D<WrongValueScalar>);
static_assert(!Basis1D<Wrong1DGradientColumns>);
static_assert(!Basis1D<MissingHessian>);
static_assert(!Basis1D<WrongHessianScalar>);
static_assert(!Basis3D<WrongHessianColumns>);
static_assert(!Basis3D<WrongHessianRows>);
static_assert(!Basis3D<WrongGradientColumns>);
static_assert(!Basis3D<WrongGradientRows>);
static_assert(!Basis3D<Wrong3DReturn>);
static_assert(!Basis1D<HexQ1Basis>);
static_assert(!Basis3D<LinearLagrangeBasis>);
static_assert(!Basis<HexQ1Basis, 1>);
static_assert(!Basis<HexQ1Basis, 2>);
static_assert(!Basis<HexQ1Basis, 0>);
static_assert(!Basis<HexQ1Basis, -1>);
static_assert(!Basis<MissingBasis, 3>);
static_assert(!Basis<NonConstantCount, 3>);

TEST(Basis1D, CoefficientInfoDistinguishesValuesAndDerivativesAtTheSamePoint) {
    const Eigen::Vector4d expected_points(0, 0, 1, 1);
    for (int i = 0; i < 4; ++i) {
        const auto info = CubicHermiteBasis::coefficient_info(i);
        EXPECT_EQ(info.point, expected_points[i]);
        EXPECT_EQ(info.derivative_order, (std::array<int, 1>{i % 2}));
        const Eigen::Vector4d sample = info.derivative_order[0] == 0 ? CubicHermiteBasis::eval_N(info.point)
                                                                     : CubicHermiteBasis::eval_dN_dxi(info.point);
        EXPECT_EQ(sample, Eigen::Vector4d::Unit(i));
    }
    for (int i = 0; i < 2; ++i) {
        const auto info = LinearLagrangeBasis::coefficient_info(i);
        EXPECT_EQ(info.point, i);
        EXPECT_EQ(info.derivative_order[0], 0);
    }
    EXPECT_THROW(CubicHermiteBasis::coefficient_info(-1), std::out_of_range);
    EXPECT_THROW(CubicHermiteBasis::coefficient_info(4), std::out_of_range);
    EXPECT_THROW(LinearLagrangeBasis::coefficient_info(-1), std::out_of_range);
    EXPECT_THROW(LinearLagrangeBasis::coefficient_info(2), std::out_of_range);
}

template <class B, int Dim> void check_value_samples() {
    for (int i = 0; i < B::num_basis_functions; ++i) {
        const auto info = B::coefficient_info(i);
        EXPECT_EQ(info.derivative_order, (std::array<int, Dim>{}));
        EXPECT_EQ(B::eval_N(info.point), (Eigen::Matrix<double, B::num_basis_functions, 1>::Unit(i)));
    }
    EXPECT_THROW(B::coefficient_info(-1), std::out_of_range);
    EXPECT_THROW(B::coefficient_info(B::num_basis_functions), std::out_of_range);
}

TEST(Basis, SimplexAndQ1CoefficientInfoSelectsNodalValues) {
    check_value_samples<TriangleP1Basis, 2>();
    check_value_samples<TetP1Basis, 3>();
    check_value_samples<HexQ1Basis, 3>();
}

TEST(Basis, HermiteBasesRejectInvalidCoefficientIndices) {
    EXPECT_THROW(QuadHermiteBasis::coefficient_info(-1), std::out_of_range);
    EXPECT_THROW(QuadHermiteBasis::coefficient_info(16), std::out_of_range);
    EXPECT_THROW(HexHermiteBasis::coefficient_info(-1), std::out_of_range);
    EXPECT_THROW(HexHermiteBasis::coefficient_info(64), std::out_of_range);
}

template <class B, int Dim> struct CountingBasis : B {
    CountingBasis() = delete;
    inline static std::array<int, 3> calls{};
    static auto eval_N(const Eigen::Matrix<double, Dim, 1>& xi) {
        ++calls[0];
        return B::eval_N(xi);
    }
    static auto eval_dN_dxi(const Eigen::Matrix<double, Dim, 1>& xi) {
        ++calls[1];
        return B::eval_dN_dxi(xi);
    }
    static auto eval_d2N_dxi2(const Eigen::Matrix<double, Dim, 1>& xi) {
        ++calls[2];
        return B::eval_d2N_dxi2(xi);
    }
};

template <class B, int Dim> void check_evaluation_counts() {
    using Counted = CountingBasis<B, Dim>;
    const Eigen::Matrix<double, Dim, 1> xi = Eigen::Matrix<double, Dim, 1>::Constant(0.3);
    auto check = [](const auto& eval, std::array<int, 3> expected) {
        Counted::calls = {};
        EXPECT_TRUE(eval().allFinite());
        EXPECT_EQ(Counted::calls, expected);
    };
    InterpolatedResponse<Counted> response(Eigen::MatrixXd::Ones(B::num_basis_functions, 3));
    check([&] { return response.eval_value(xi); }, {1, 0, 0});
    check([&] { return Eigen::MatrixXd(response.eval_jacobian(xi)); }, {1, 1, 0});
    check([&] { return Eigen::MatrixXd(response.eval_hessian(xi, Vec::Ones(3))); }, {0, 0, 1});
    const Eigen::MatrixXd points = xi.transpose();
    check(
        [&] {
            auto r = value_sampling_response<Counted>(points, 3);
            return r.eval_value(Vec::Ones(3 * B::num_basis_functions));
        },
        {1, 0, 0});
    check(
        [&] {
            auto r = gradient_sampling_response<Counted>(points, 3);
            return r.eval_value(Vec::Ones(3 * B::num_basis_functions));
        },
        {0, 1, 0});
    check(
        [&] {
            auto r = hessian_sampling_response<Counted>(points, 3);
            return r.eval_value(Vec::Ones(3 * B::num_basis_functions));
        },
        {0, 0, 1});
}

TEST(Basis2D, FieldsEvaluateOnlyRequestedQuantities) {
    check_evaluation_counts<QuadHermiteBasis, 2>();
}

TEST(Basis3D, FieldsEvaluateOnlyRequestedQuantities) {
    check_evaluation_counts<HexQ1Basis, 3>();
    check_evaluation_counts<HexHermiteBasis, 3>();
}
} // namespace

} // namespace litepgo::fem::test
