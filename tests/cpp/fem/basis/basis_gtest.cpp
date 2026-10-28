#include "litepgo/fem/basis/1d/cubic_hermite_basis.hpp"
#include "litepgo/fem/basis/1d/linear_lagrange_basis.hpp"
#include "litepgo/fem/basis/2d/quad_hermite_basis.hpp"
#include "litepgo/fem/basis/2d/tensor_product_basis_2d.hpp"
#include "litepgo/fem/basis/3d/hex_hermite_basis.hpp"
#include "litepgo/fem/basis/3d/hex_q1_basis.hpp"
#include "litepgo/fem/basis/3d/tensor_product_basis_3d.hpp"
#include "litepgo/fem/basis/3d/tet_p1_basis.hpp"
#include "litepgo/fem/basis/basis.hpp"

namespace {

static_assert(std::same_as<litepgo::LinearLagrangeBasis::Sample, litepgo::BasisSample<2, 1>>);
static_assert(std::same_as<litepgo::CubicHermiteBasis::Sample, litepgo::BasisSample<4, 1>>);
static_assert(std::same_as<litepgo::QuadHermiteBasis::Sample, litepgo::BasisSample<16, 2>>);
static_assert(std::same_as<litepgo::TetP1Basis::Sample, litepgo::BasisSample<4, 3>>);
static_assert(std::same_as<litepgo::HexQ1Basis::Sample, litepgo::BasisSample<8, 3>>);
static_assert(std::same_as<litepgo::HexHermiteBasis::Sample, litepgo::BasisSample<64, 3>>);

struct ConstantBasis {
    static constexpr int num_basis_functions = 1;
    using Sample = litepgo::BasisSample<num_basis_functions, 1>;
    static Sample evaluate(double) {
        return {Eigen::Matrix<double, 1, 1>(1.0), Eigen::Matrix<double, 1, 1>(0.0), Eigen::Matrix<double, 1, 1>(0.0)};
    }
};

struct InstanceOnly1D : ConstantBasis {
    Sample evaluate(double) const;
};
struct InstanceOnly3D {
    static constexpr int num_basis_functions = 8;
    using Sample = litepgo::HexQ1Basis::Sample;
    Sample evaluate(const Eigen::Vector3d&) const;
};
static_assert(!litepgo::Basis1D<InstanceOnly1D>);
static_assert(!litepgo::Basis3D<InstanceOnly3D>);

struct MissingSample {};

struct MissingEvaluate {
    static constexpr int num_basis_functions = 1;
    using Sample = ConstantBasis::Sample;
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
    static double evaluate(double);
};

struct WrongGradientSize : ConstantBasis {
    struct Sample : litepgo::BasisSample<1, 1> {
        Eigen::Matrix<double, 2, 1> dN_dxi;
    };
    static Sample evaluate(double);
};

struct WrongValueScalar : ConstantBasis {
    struct Sample : litepgo::BasisSample<1, 1> {
        Eigen::Matrix<float, 1, 1> N;
    };
    static Sample evaluate(double);
};

template <class B>
concept CanFormTensorProduct = requires { typename litepgo::TensorProductBasis3D<B>; };

template <class B>
concept CanFormTensorProduct2D = requires { typename litepgo::TensorProductBasis2D<B>; };
static_assert(CanFormTensorProduct2D<litepgo::LinearLagrangeBasis>);
static_assert(CanFormTensorProduct2D<ConstantBasis>);

static_assert(litepgo::Basis1D<litepgo::LinearLagrangeBasis>);
static_assert(litepgo::Basis1D<ConstantBasis>);
static_assert(!litepgo::Basis1D<MissingSample>);
static_assert(!litepgo::Basis1D<MissingEvaluate>);
static_assert(!litepgo::Basis1D<NonConstantCount>);
static_assert(!litepgo::Basis1D<ZeroCount>);
static_assert(!litepgo::Basis1D<NegativeCount>);
static_assert(!litepgo::Basis1D<WrongReturn>);
static_assert(!litepgo::Basis1D<WrongGradientSize>);
static_assert(!litepgo::Basis1D<WrongValueScalar>);

static_assert(CanFormTensorProduct<litepgo::LinearLagrangeBasis>);
static_assert(CanFormTensorProduct<ConstantBasis>);
static_assert(!CanFormTensorProduct<WrongReturn>);
static_assert(!CanFormTensorProduct<WrongGradientSize>);

struct Wrong1DGradientColumns : litepgo::LinearLagrangeBasis {
    struct Sample : litepgo::BasisSample<2, 1> {
        Eigen::Matrix<double, 2, 2> dN_dxi;
    };
    static Sample evaluate(double);
};

struct WrongGradientColumns : litepgo::TensorProductBasis3D<litepgo::LinearLagrangeBasis> {
    struct Sample : litepgo::BasisSample<8, 3> {
        Eigen::Matrix<double, 8, 2> dN_dxi;
    };
    static Sample evaluate(const Eigen::Vector3d&);
};

struct WrongGradientRows : litepgo::TensorProductBasis3D<litepgo::LinearLagrangeBasis> {
    struct Sample : litepgo::BasisSample<8, 3> {
        Eigen::Matrix<double, 7, 3> dN_dxi;
    };
    static Sample evaluate(const Eigen::Vector3d&);
};

struct Wrong3DReturn : litepgo::TensorProductBasis3D<litepgo::LinearLagrangeBasis> {
    static double evaluate(const Eigen::Vector3d&);
};

struct MissingHessian : ConstantBasis {
    struct Sample {
        Eigen::Matrix<double, 1, 1> N;
        Eigen::Matrix<double, 1, 1> dN_dxi;
    };
    static Sample evaluate(double);
};
struct WrongHessianScalar : ConstantBasis {
    struct Sample : litepgo::BasisSample<1, 1> {
        Eigen::Matrix<float, 1, 1> d2N_dxi2;
    };
    static Sample evaluate(double);
};
struct WrongHessianColumns {
    static constexpr int num_basis_functions = 8;
    struct Sample : litepgo::BasisSample<8, 3> {
        Eigen::Matrix<double, 8, 5> d2N_dxi2;
    };
    static Sample evaluate(const Eigen::Vector3d&);
};
struct WrongHessianRows {
    static constexpr int num_basis_functions = 8;
    struct Sample : litepgo::BasisSample<8, 3> {
        Eigen::Matrix<double, 7, 6> d2N_dxi2;
    };
    static Sample evaluate(const Eigen::Vector3d&);
};
static_assert(!litepgo::Basis1D<MissingHessian>);
static_assert(!CanFormTensorProduct<MissingHessian>);
static_assert(!CanFormTensorProduct2D<MissingHessian>);
static_assert(!litepgo::Basis1D<WrongHessianScalar>);
static_assert(!litepgo::Basis3D<WrongHessianColumns>);
static_assert(!litepgo::Basis3D<WrongHessianRows>);

static_assert(litepgo::Basis<litepgo::LinearLagrangeBasis, 1>);
static_assert(litepgo::Basis<litepgo::TensorProductBasis3D<litepgo::LinearLagrangeBasis>, 3>);
static_assert(litepgo::Basis3D<litepgo::TensorProductBasis3D<litepgo::LinearLagrangeBasis>>);
static_assert(litepgo::Basis3D<litepgo::HexQ1Basis>);
static_assert(!litepgo::Basis1D<litepgo::HexQ1Basis>);
static_assert(litepgo::Basis3D<litepgo::TensorProductBasis3D<ConstantBasis>>);
static_assert(!litepgo::Basis1D<Wrong1DGradientColumns>);
static_assert(!litepgo::Basis3D<litepgo::LinearLagrangeBasis>);
static_assert(!litepgo::Basis3D<WrongGradientColumns>);
static_assert(!litepgo::Basis3D<WrongGradientRows>);
static_assert(!litepgo::Basis3D<Wrong3DReturn>);
static_assert(!litepgo::Basis<litepgo::TensorProductBasis3D<litepgo::LinearLagrangeBasis>, 1>);
static_assert(!litepgo::Basis<litepgo::LinearLagrangeBasis, 3>);
static_assert(!litepgo::Basis<litepgo::TensorProductBasis3D<litepgo::LinearLagrangeBasis>, 2>);
static_assert(!litepgo::Basis<litepgo::TensorProductBasis3D<litepgo::LinearLagrangeBasis>, 0>);
static_assert(!litepgo::Basis<litepgo::TensorProductBasis3D<litepgo::LinearLagrangeBasis>, -1>);
static_assert(!litepgo::Basis<MissingSample, 3>);
static_assert(!litepgo::Basis<NonConstantCount, 3>);
static_assert(!litepgo::Basis<WrongGradientColumns, 3>);
static_assert(!litepgo::Basis<WrongGradientRows, 3>);
static_assert(!litepgo::Basis<Wrong3DReturn, 3>);

} // namespace
