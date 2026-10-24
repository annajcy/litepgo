#include "litepgo/fem/volumetric_material.hpp"

namespace {

using namespace litepgo;

template <int N> struct BaseMaterial {
    static constexpr int num_parameters = N;
    using Parameters = typename MaterialTypes<N>::Parameters;
    double value(const Mat3&, const Mat3&, const Parameters&) const;
    Mat3 gradient_F(const Mat3&, const Mat3&, const Parameters&) const;
    Mat9 hessian_F(const Mat3&, const Mat3&, const Parameters&) const;
};

template <int Mask> struct OptionalMaterial : BaseMaterial<2> {
    Mat3 gradient_Fp(const Mat3&, const Mat3&, const Parameters&) const
        requires((Mask & 1) != 0);
    Mat9 hessian_Fp(const Mat3&, const Mat3&, const Parameters&) const
        requires((Mask & 2) != 0);
    Parameters gradient_theta(const Mat3&, const Mat3&, const Parameters&) const
        requires((Mask & 4) != 0);
    Eigen::Matrix2d hessian_theta(const Mat3&, const Mat3&, const Parameters&) const
        requires((Mask & 8) != 0);
};

template <int Mask>
constexpr bool capabilities_match = VolumetricMaterial<OptionalMaterial<Mask>> &&
                                    (MaterialWithFpGradient<OptionalMaterial<Mask>> == ((Mask & 1) != 0)) &&
                                    (MaterialWithFpHessian<OptionalMaterial<Mask>> == ((Mask & 2) != 0)) &&
                                    (MaterialWithThetaGradient<OptionalMaterial<Mask>> == ((Mask & 4) != 0)) &&
                                    (MaterialWithThetaHessian<OptionalMaterial<Mask>> == ((Mask & 8) != 0));

static_assert(capabilities_match<0> && capabilities_match<1> && capabilities_match<2> && capabilities_match<3>);
static_assert(capabilities_match<4> && capabilities_match<5> && capabilities_match<6> && capabilities_match<7>);
static_assert(capabilities_match<8> && capabilities_match<9> && capabilities_match<10> && capabilities_match<11>);
static_assert(capabilities_match<12> && capabilities_match<13> && capabilities_match<14> && capabilities_match<15>);
static_assert(VolumetricMaterial<BaseMaterial<0>>);
static_assert(VolumetricMaterial<BaseMaterial<1>>);
static_assert(VolumetricMaterial<BaseMaterial<2>>);

struct MissingMetadata {};
struct NonConstantCount : BaseMaterial<2> {
    inline static int num_parameters = 2;
};
struct NegativeCount : BaseMaterial<2> {
    static constexpr int num_parameters = -1;
};
struct WrongParameters : BaseMaterial<2> {
    using Parameters = Eigen::Vector3d;
};
struct DynamicParameters : BaseMaterial<2> {
    using Parameters = Eigen::VectorXd;
};
struct MissingValue : BaseMaterial<2> {
    double value(const Mat3&, const Mat3&, const Parameters&) const = delete;
};
struct MissingGradient : BaseMaterial<2> {
    Mat3 gradient_F(const Mat3&, const Mat3&, const Parameters&) const = delete;
};
struct MissingHessian : BaseMaterial<2> {
    Mat9 hessian_F(const Mat3&, const Mat3&, const Parameters&) const = delete;
};
struct WrongValue : BaseMaterial<2> {
    float value(const Mat3&, const Mat3&, const Parameters&) const;
};
struct WrongGradient : BaseMaterial<2> {
    Eigen::Vector3d gradient_F(const Mat3&, const Mat3&, const Parameters&) const;
};
struct WrongHessian : BaseMaterial<2> {
    Mat3 hessian_F(const Mat3&, const Mat3&, const Parameters&) const;
};
struct MutableMaterial : BaseMaterial<2> {
    double value(const Mat3&, const Mat3&, const Parameters&);
};
struct MutableInput : BaseMaterial<2> {
    double value(Mat3&, const Mat3&, const Parameters&) const;
};

static_assert(!VolumetricMaterial<MissingMetadata>);
static_assert(!VolumetricMaterial<NonConstantCount>);
static_assert(!VolumetricMaterial<NegativeCount>);
static_assert(!VolumetricMaterial<WrongParameters>);
static_assert(!VolumetricMaterial<DynamicParameters>);
static_assert(!VolumetricMaterial<MissingValue>);
static_assert(!VolumetricMaterial<MissingGradient>);
static_assert(!VolumetricMaterial<MissingHessian>);
static_assert(!VolumetricMaterial<WrongValue>);
static_assert(!VolumetricMaterial<WrongGradient>);
static_assert(!VolumetricMaterial<WrongHessian>);
static_assert(!VolumetricMaterial<MutableMaterial>);
static_assert(!VolumetricMaterial<MutableInput>);
static_assert(!MaterialWithFpGradient<MissingMetadata>);
static_assert(!MaterialWithFpHessian<MissingMetadata>);
static_assert(!MaterialWithThetaGradient<MissingMetadata>);
static_assert(!MaterialWithThetaHessian<MissingMetadata>);

struct WrongFpGradient : OptionalMaterial<15> {
    Mat9 gradient_Fp(const Mat3&, const Mat3&, const Parameters&) const;
};
struct WrongFpHessian : OptionalMaterial<15> {
    Mat3 hessian_Fp(const Mat3&, const Mat3&, const Parameters&) const;
};
struct WrongThetaGradient : OptionalMaterial<15> {
    Eigen::Vector3d gradient_theta(const Mat3&, const Mat3&, const Parameters&) const;
};
struct WrongThetaHessian : OptionalMaterial<15> {
    Eigen::Matrix3d hessian_theta(const Mat3&, const Mat3&, const Parameters&) const;
};
struct MutableFpGradient : BaseMaterial<2> {
    Mat3 gradient_Fp(const Mat3&, const Mat3&, const Parameters&);
};
struct MutableFpHessian : BaseMaterial<2> {
    Mat9 hessian_Fp(const Mat3&, const Mat3&, const Parameters&);
};
struct MutableThetaGradient : BaseMaterial<2> {
    Parameters gradient_theta(const Mat3&, const Mat3&, const Parameters&);
};
struct MutableThetaHessian : BaseMaterial<2> {
    Eigen::Matrix2d hessian_theta(const Mat3&, const Mat3&, const Parameters&);
};

static_assert(VolumetricMaterial<WrongFpGradient>);
static_assert(!MaterialWithFpGradient<WrongFpGradient>);
static_assert(MaterialWithFpHessian<WrongFpGradient>);
static_assert(!MaterialWithFpHessian<WrongFpHessian>);
static_assert(MaterialWithFpGradient<WrongFpHessian>);
static_assert(!MaterialWithThetaGradient<WrongThetaGradient>);
static_assert(MaterialWithThetaHessian<WrongThetaGradient>);
static_assert(!MaterialWithThetaHessian<WrongThetaHessian>);
static_assert(MaterialWithThetaGradient<WrongThetaHessian>);
static_assert(!MaterialWithFpGradient<MutableFpGradient>);
static_assert(!MaterialWithFpHessian<MutableFpHessian>);
static_assert(!MaterialWithThetaGradient<MutableThetaGradient>);
static_assert(!MaterialWithThetaHessian<MutableThetaHessian>);

} // namespace
