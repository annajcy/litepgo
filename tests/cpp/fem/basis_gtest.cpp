#include "litepgo/fem/basis.h"
#include "litepgo/fem/hex_q1_basis.h"
#include "litepgo/fem/lagrange_basis.h"
#include "litepgo/fem/tensor_product_basis.h"

namespace {

struct ConstantBasis {
    static constexpr int num_basis_functions = 1;
    struct Sample {
        Eigen::Matrix<double, 1, 1> values;
        Eigen::Matrix<double, 1, 1> gradients;
    };
    static Sample evaluate(double) {
        return {Eigen::Matrix<double, 1, 1>(1.0), Eigen::Matrix<double, 1, 1>(0.0)};
    }
};

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
    struct Sample {
        Eigen::Matrix<double, 1, 1> values;
        Eigen::Matrix<double, 2, 1> gradients;
    };
    static Sample evaluate(double);
};

struct WrongValueScalar : ConstantBasis {
    struct Sample {
        Eigen::Matrix<float, 1, 1> values;
        Eigen::Matrix<double, 1, 1> gradients;
    };
    static Sample evaluate(double);
};

template <class B>
concept CanFormTensorProduct = requires { typename litepgo::TensorProductBasis3D<B>; };

static_assert(litepgo::Basis1D<litepgo::LagrangeBasis>);
static_assert(litepgo::Basis1D<ConstantBasis>);
static_assert(!litepgo::Basis1D<MissingSample>);
static_assert(!litepgo::Basis1D<MissingEvaluate>);
static_assert(!litepgo::Basis1D<NonConstantCount>);
static_assert(!litepgo::Basis1D<ZeroCount>);
static_assert(!litepgo::Basis1D<NegativeCount>);
static_assert(!litepgo::Basis1D<WrongReturn>);
static_assert(!litepgo::Basis1D<WrongGradientSize>);
static_assert(!litepgo::Basis1D<WrongValueScalar>);

static_assert(CanFormTensorProduct<litepgo::LagrangeBasis>);
static_assert(CanFormTensorProduct<ConstantBasis>);
static_assert(!CanFormTensorProduct<WrongReturn>);
static_assert(!CanFormTensorProduct<WrongGradientSize>);

using Q1 = litepgo::TensorProductBasis3D<litepgo::LagrangeBasis>;

struct Wrong1DGradientColumns : litepgo::LagrangeBasis {
    struct Sample {
        Eigen::Matrix<double, 2, 1> values;
        Eigen::Matrix<double, 2, 2> gradients;
    };
    static Sample evaluate(double);
};

struct WrongGradientColumns : Q1 {
    struct Sample {
        Eigen::Matrix<double, 8, 1> values;
        Eigen::Matrix<double, 8, 2> gradients;
    };
    static Sample evaluate(const Eigen::Vector3d&);
};

struct WrongGradientRows : Q1 {
    struct Sample {
        Eigen::Matrix<double, 8, 1> values;
        Eigen::Matrix<double, 7, 3> gradients;
    };
    static Sample evaluate(const Eigen::Vector3d&);
};

struct Wrong3DReturn : Q1 {
    static double evaluate(const Eigen::Vector3d&);
};

static_assert(litepgo::Basis<litepgo::LagrangeBasis, 1>);
static_assert(litepgo::Basis<Q1, 3>);
static_assert(litepgo::Basis3D<Q1>);
static_assert(litepgo::Basis3D<litepgo::HexQ1Basis>);
static_assert(!litepgo::Basis1D<litepgo::HexQ1Basis>);
static_assert(litepgo::Basis3D<litepgo::TensorProductBasis3D<ConstantBasis>>);
static_assert(!litepgo::Basis1D<Wrong1DGradientColumns>);
static_assert(!litepgo::Basis3D<litepgo::LagrangeBasis>);
static_assert(!litepgo::Basis3D<WrongGradientColumns>);
static_assert(!litepgo::Basis3D<WrongGradientRows>);
static_assert(!litepgo::Basis3D<Wrong3DReturn>);
static_assert(!litepgo::Basis<Q1, 1>);
static_assert(!litepgo::Basis<litepgo::LagrangeBasis, 3>);
static_assert(!litepgo::Basis<Q1, 2>);
static_assert(!litepgo::Basis<Q1, 0>);
static_assert(!litepgo::Basis<Q1, -1>);
static_assert(!litepgo::Basis<MissingSample, 3>);
static_assert(!litepgo::Basis<NonConstantCount, 3>);
static_assert(!litepgo::Basis<WrongGradientColumns, 3>);
static_assert(!litepgo::Basis<WrongGradientRows, 3>);
static_assert(!litepgo::Basis<Wrong3DReturn, 3>);

} // namespace
