#include "litepgo/fem/basis/1d/cubic_hermite_basis.hpp"
#include "litepgo/fem/basis/1d/linear_lagrange_basis.hpp"
#include "litepgo/fem/basis/2d/quad_hermite_basis.hpp"
#include "litepgo/fem/basis/2d/tensor_product_basis_2d.hpp"
#include "litepgo/fem/basis/2d/triangle_p1_basis.hpp"
#include "litepgo/fem/basis/3d/hex_hermite_basis.hpp"
#include "litepgo/fem/basis/3d/hex_q1_basis.hpp"
#include "litepgo/fem/basis/3d/tensor_product_basis_3d.hpp"
#include "litepgo/fem/basis/3d/tet_p1_basis.hpp"
#include "litepgo/fem/basis/basis.hpp"
#include "litepgo/fem/field/interpolated_field.hpp"

#include <gtest/gtest.h>

#include <array>
#include <limits>

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
struct InstanceOnly3D : TensorProductBasis3D<LinearLagrangeBasis> {
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
struct WrongGradientColumns : TensorProductBasis3D<LinearLagrangeBasis> {
    static Eigen::Matrix<double, 8, 2> eval_dN_dxi(const Eigen::Vector3d&);
};
struct WrongGradientRows : TensorProductBasis3D<LinearLagrangeBasis> {
    static Eigen::Matrix<double, 7, 3> eval_dN_dxi(const Eigen::Vector3d&);
};
struct Wrong3DReturn : TensorProductBasis3D<LinearLagrangeBasis> {
    static double eval_N(const Eigen::Vector3d&);
};
struct MissingHessian : ConstantBasis {
    static Eigen::Matrix<double, 1, 1> eval_d2N_dxi2(double) = delete;
};
struct WrongHessianScalar : ConstantBasis {
    static Eigen::Matrix<float, 1, 1> eval_d2N_dxi2(double);
};
struct WrongHessianColumns : TensorProductBasis3D<LinearLagrangeBasis> {
    static Eigen::Matrix<double, 8, 5> eval_d2N_dxi2(const Eigen::Vector3d&);
};
struct WrongHessianRows : TensorProductBasis3D<LinearLagrangeBasis> {
    static Eigen::Matrix<double, 7, 6> eval_d2N_dxi2(const Eigen::Vector3d&);
};

template <class B>
concept CanFormTensorProduct = requires { typename TensorProductBasis3D<B>; };
template <class B>
concept CanFormTensorProduct2D = requires { typename TensorProductBasis2D<B>; };

static_assert(Basis1D<LinearLagrangeBasis>);
static_assert(Basis1D<CubicHermiteBasis>);
static_assert(Basis1D<ConstantBasis>);
static_assert(Basis2D<QuadHermiteBasis>);
static_assert(Basis2D<TensorProductBasis2D<ConstantBasis>>);
static_assert(Basis3D<HexQ1Basis>);
static_assert(Basis3D<HexHermiteBasis>);
static_assert(Basis3D<TetP1Basis>);
static_assert(std::same_as<HexQ1Basis, TensorProductBasis3D<LinearLagrangeBasis>>);
static_assert(std::same_as<QuadHermiteBasis, TensorProductBasis2D<CubicHermiteBasis>>);
static_assert(std::same_as<HexHermiteBasis, TensorProductBasis3D<CubicHermiteBasis>>);
static_assert(Basis3D<TensorProductBasis3D<ConstantBasis>>);
static_assert(Basis<TensorProductBasis3D<LinearLagrangeBasis>, 3>);
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
static_assert(!Basis<TensorProductBasis3D<LinearLagrangeBasis>, 1>);
static_assert(!Basis<TensorProductBasis3D<LinearLagrangeBasis>, 2>);
static_assert(!Basis<TensorProductBasis3D<LinearLagrangeBasis>, 0>);
static_assert(!Basis<TensorProductBasis3D<LinearLagrangeBasis>, -1>);
static_assert(!Basis<MissingBasis, 3>);
static_assert(!Basis<NonConstantCount, 3>);
static_assert(CanFormTensorProduct<LinearLagrangeBasis>);
static_assert(CanFormTensorProduct<ConstantBasis>);
static_assert(CanFormTensorProduct2D<LinearLagrangeBasis>);
static_assert(CanFormTensorProduct2D<ConstantBasis>);
static_assert(!CanFormTensorProduct<WrongReturn>);
static_assert(!CanFormTensorProduct<WrongGradientSize>);
static_assert(!CanFormTensorProduct<MissingHessian>);
static_assert(!CanFormTensorProduct2D<MissingHessian>);

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

template <bool BadPoint> struct InvalidCoefficientInfo : LinearLagrangeBasis {
    static CoefficientInfo<1> coefficient_info(int i) {
        auto info = LinearLagrangeBasis::coefficient_info(i);
        if constexpr (BadPoint)
            info.point = std::numeric_limits<double>::quiet_NaN();
        else
            info.derivative_order[0] = -1;
        return info;
    }
};

TEST(Basis, TensorProductsRejectInvalidSampleDescriptorsBeforeOrdering) {
    EXPECT_THROW((TensorProductBasis2D<InvalidCoefficientInfo<true>>::eval_N(Eigen::Vector2d::Zero())),
                 std::invalid_argument);
    EXPECT_THROW((TensorProductBasis3D<InvalidCoefficientInfo<false>>::coefficient_info(0)), std::invalid_argument);
    EXPECT_THROW(QuadHermiteBasis::coefficient_info(-1), std::out_of_range);
    EXPECT_THROW(QuadHermiteBasis::coefficient_info(16), std::out_of_range);
    EXPECT_THROW(HexHermiteBasis::coefficient_info(-1), std::out_of_range);
    EXPECT_THROW(HexHermiteBasis::coefficient_info(64), std::out_of_range);
}

struct CountingBasis : LinearLagrangeBasis {
    CountingBasis() = delete;
    inline static std::array<int, 3> calls{};
    static Eigen::Vector2d eval_N(double t) {
        ++calls[0];
        return LinearLagrangeBasis::eval_N(t);
    }
    static Eigen::Vector2d eval_dN_dxi(double t) {
        ++calls[1];
        return LinearLagrangeBasis::eval_dN_dxi(t);
    }
    static Eigen::Vector2d eval_d2N_dxi2(double t) {
        ++calls[2];
        return LinearLagrangeBasis::eval_d2N_dxi2(t);
    }
};

template <class B, int Dim> void check_evaluation_counts() {
    const Eigen::Matrix<double, Dim, 1> xi = Eigen::Matrix<double, Dim, 1>::Constant(0.3);
    auto check = [](const auto& eval, std::array<int, 3> expected) {
        CountingBasis::calls = {};
        EXPECT_TRUE(eval().allFinite());
        EXPECT_EQ(CountingBasis::calls, expected);
    };
    check([&] { return B::eval_N(xi); }, {Dim, 0, 0});
    check([&] { return B::eval_dN_dxi(xi); }, {Dim, Dim, 0});
    check([&] { return B::eval_d2N_dxi2(xi); }, {Dim, Dim, Dim});

    const InterpolatedFieldT<B, Dim> field(Eigen::MatrixXd::Ones(B::num_basis_functions, 3));
    check([&] { return field.eval_value(xi); }, {Dim, 0, 0});
    check([&] { return field.eval_dvalue_dcoeffs(xi); }, {Dim, 0, 0});
    check([&] { return field.eval_dvalue_dxi(xi); }, {Dim, Dim, 0});
    check([&] { return field.eval_dgrad_xi_dcoeffs(xi); }, {Dim, Dim, 0});
    check([&] { return field.eval_d2value_dxi2(xi); }, {Dim, Dim, Dim});
    check([&] { return field.eval_dhess_xi_dcoeffs(xi); }, {Dim, Dim, Dim});
}

TEST(Basis2D, TensorProductsAndFieldsEvaluateOnlyRequestedQuantities) {
    check_evaluation_counts<TensorProductBasis2D<CountingBasis>, 2>();
}

TEST(Basis3D, TensorProductsAndFieldsEvaluateOnlyRequestedQuantities) {
    check_evaluation_counts<TensorProductBasis3D<CountingBasis>, 3>();
}
} // namespace

} // namespace litepgo::fem::test
