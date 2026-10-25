#include "litepgo/dof/constant_dof_map.hpp"
#include "litepgo/dof/element_dof_map.hpp"
#include "litepgo/dof/vertex_dof_map.hpp"
#include "litepgo/energy/mesh_energy.hpp"
#include "litepgo/fem/basis/hex_q1_basis.hpp"
#include "litepgo/fem/field/deformation_gradient_field.hpp"
#include "litepgo/fem/field/identity_field.hpp"
#include "litepgo/fem/field/interpolated_value_field.hpp"
#include "litepgo/fem/local/integrate_volume.hpp"
#include "litepgo/fem/local/material_local_energy.hpp"
#include "litepgo/fem/material/neo_hookean.hpp"
#include "litepgo/fem/quadrature/gauss_legendre_3d.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <tbb/global_control.h>

namespace {
using namespace litepgo;

HexTopology two_hexes() {
    return HexTopology(12, {{1, 3, 9, 4, 7, 6, 0, 10}, {3, 8, 5, 9, 6, 2, 11, 0}});
}
Interpolation<HexQ1Basis> geometry() {
    Eigen::Matrix<double, 8, 3> nodes;
    for (int i = 0; i < 8; ++i)
        nodes.row(i) = HexQ1Basis::reference_node(i).transpose();
    return Interpolation<HexQ1Basis>(nodes);
}
MaterialFields passive_fields() {
    Eigen::Matrix3d F;
    F << 1.1, 0.04, 0.02, -0.03, 0.95, 0.01, 0.02, 0.03, 1.05;
    return {std::make_shared<IdentityField<Eigen::Matrix3d>>(F),
            std::make_shared<IdentityField<Eigen::Matrix3d>>(Eigen::Matrix3d::Identity()),
            std::make_shared<IdentityField<Eigen::VectorXd>>(Eigen::Vector2d(2, 3))};
}

void check_fd(MeshEnergy& energy, const Vec& x) {
    Vec gradient;
    SpMat hessian = energy.hessian_pattern();
    const double value = energy.evaluate(x, &gradient, &hessian);
    EXPECT_DOUBLE_EQ(value, energy.value(x));
    EXPECT_TRUE(gradient.isApprox(energy.gradient(x), 1e-13));
    EXPECT_TRUE(Eigen::MatrixXd(hessian).isApprox(Eigen::MatrixXd(energy.hessian(x)), 1e-13));
    const double step = 1e-5;
    for (Eigen::Index j = 0; j < x.size(); ++j) {
        Vec plus = x, minus = x;
        plus[j] += step;
        minus[j] -= step;
        EXPECT_NEAR(gradient[j], (energy.value(plus) - energy.value(minus)) / (2 * step), 1e-7);
        const Vec fd = (energy.gradient(plus) - energy.gradient(minus)) / (2 * step);
        EXPECT_LT((fd - Eigen::MatrixXd(hessian).col(j)).norm(), 1e-6);
    }
    {
        tbb::global_control threads(tbb::global_control::max_allowed_parallelism, 1);
        Vec g;
        SpMat h = energy.hessian_pattern();
        EXPECT_DOUBLE_EQ(energy.evaluate(x, &g, &h), value);
        EXPECT_EQ(g, gradient);
        EXPECT_EQ(Eigen::MatrixXd(h), Eigen::MatrixXd(hessian));
    }
    {
        tbb::global_control threads(tbb::global_control::max_allowed_parallelism, 4);
        Vec g;
        SpMat h = energy.hessian_pattern();
        EXPECT_DOUBLE_EQ(energy.evaluate(x, &g, &h), value);
        EXPECT_EQ(g, gradient);
        EXPECT_EQ(Eigen::MatrixXd(h), Eigen::MatrixXd(hessian));
    }
    EXPECT_FALSE(energy.update_pattern(x));
}

template <class Map, class Fields, class Bind>
void check_material_path(const Map& map, const Vec& data, Fields fields, Bind bind) {
    const auto reference = geometry();
    MeshEnergy energy(map, [reference, fields, bind](int, const auto& local_dof_values, Vec* g, Eigen::MatrixXd* h) {
        return integrate_volume<GaussLegendre3D<2>>(reference, bind(fields(local_dof_values)), g, h);
    });
    // This test only evaluates nearby feasible states; it does not use the default step for a solver.
    check_fd(energy, data);
}

TEST(MeshEnergy, HexNodeDisplacementAndPlasticDisplacement) {
    const auto topology = two_hexes();
    const VertexDofMap map(topology, 3);
    Vec data = Vec::Zero(36);
    for (int i = 0; i < data.size(); ++i)
        data[i] = 0.01 * std::sin(i + 0.3);
    check_material_path(
        map, data,
        [](const auto& local_dof_values) {
            EXPECT_EQ(local_dof_values.rows(), 8);
            EXPECT_EQ(local_dof_values.cols(), 3);
            auto fields = passive_fields();
            fields.F = std::make_shared<DeformationGradientField<litepgo::HexQ1Basis>>(local_dof_values);
            return fields;
        },
        [](auto fields) { return MaterialLocalEnergyF<NeoHookean>(std::move(fields)); });
    check_material_path(
        map, data,
        [](const auto& local_dof_values) {
            auto fields = passive_fields();
            fields.Fp = std::make_shared<DeformationGradientField<litepgo::HexQ1Basis>>(local_dof_values);
            return fields;
        },
        [](auto fields) { return MaterialLocalEnergyFp<NeoHookean>(std::move(fields)); });
}

TEST(MeshEnergy, HexElementPlasticAndNodeOrSharedParameters) {
    const auto topology = two_hexes();
    Vec plastic = Vec::Zero(18);
    for (int e = 0; e < 2; ++e) {
        Eigen::Matrix3d p = Eigen::Matrix3d::Identity();
        p(0, 1) = 0.03;
        p(2, 2) += 0.04 * e;
        plastic.segment<9>(9 * e) = p.reshaped();
    }
    check_material_path(
        ElementDofMap(topology, 9), plastic,
        [](const Eigen::VectorXd& local_dof_values) {
            auto fields = passive_fields();
            fields.Fp = std::make_shared<IdentityField<Eigen::Matrix3d>>(local_dof_values.reshaped(3, 3));
            return fields;
        },
        [](auto fields) { return MaterialLocalEnergyFp<NeoHookean>(std::move(fields)); });
    Vec nodal = Vec::Zero(24);
    for (int i = 0; i < 12; ++i) {
        nodal[2 * i] = 2 + 0.01 * i;
        nodal[2 * i + 1] = 3 + 0.02 * i;
    }
    check_material_path(
        VertexDofMap(topology, 2), nodal,
        [](const auto& local_dof_values) {
            auto fields = passive_fields();
            fields.theta = std::make_shared<InterpolatedValueField<litepgo::HexQ1Basis>>(local_dof_values);
            return fields;
        },
        [](auto fields) { return MaterialLocalEnergyTheta<NeoHookean>(std::move(fields)); });
    Vec shared(Eigen::Vector2d(2, 3));
    check_material_path(
        ConstantDofMap(topology, 2), shared,
        [](const Eigen::VectorXd& local_dof_values) {
            auto fields = passive_fields();
            fields.theta = std::make_shared<IdentityField<Eigen::VectorXd>>(local_dof_values);
            return fields;
        },
        [](auto fields) { return MaterialLocalEnergyTheta<NeoHookean>(std::move(fields)); });
}

TEST(MeshEnergy, OwnsPlanUsesCurrentXAndSupportsTriangleAndTet) {
    const auto quadratic = [](int e, const auto& local_dof_values, Vec* g, Eigen::MatrixXd* h) {
        const Vec flat = local_dof_values.template reshaped<Eigen::RowMajor>();
        if (g)
            *g = (e + 1) * flat;
        if (h)
            *h = (e + 1) * Eigen::MatrixXd::Identity(flat.size(), flat.size());
        return 0.5 * (e + 1) * flat.squaredNorm();
    };
    MeshEnergy triangle(VertexDofMap(TriangleTopology(4, {{2, 0, 1}, {1, 0, 3}}), 2), quadratic);
    Vec x = Vec::LinSpaced(8, 0.1, 0.8);
    check_fd(triangle, x);
    Vec counts(8);
    counts << 3, 3, 3, 3, 1, 1, 2, 2;
    EXPECT_TRUE(triangle.gradient(x).isApprox(counts.cwiseProduct(x), 1e-14));
    EXPECT_TRUE(Eigen::MatrixXd(triangle.hessian(x)).isApprox(Eigen::MatrixXd(counts.asDiagonal()), 1e-14));
    EXPECT_NE(triangle.value(x), triangle.value(2 * x));
    EXPECT_DOUBLE_EQ(triangle.max_feasible_step(x, x), 1);
    MeshEnergy tet(VertexDofMap(TetTopology(4, {{3, 1, 2, 0}}), 1), quadratic);
    check_fd(tet, Vec::LinSpaced(4, 0.1, 0.4));
}

TEST(MeshEnergy, ProjectsEachLocalHessianBeforeScatter) {
    const auto topology = two_hexes();
    MeshEnergy energy(ConstantDofMap(topology, 1),
                      [](int e, const Eigen::VectorXd& local_dof_values, Vec* g, Eigen::MatrixXd* h) {
                          double k = e == 0 ? -2 : 3;
                          if (g)
                              *g = k * local_dof_values;
                          if (h)
                              *h = Eigen::MatrixXd::Constant(1, 1, k);
                          return 0.5 * k * local_dof_values.squaredNorm();
                      });
    EXPECT_EQ(energy.hessian(Vec::Ones(1)).coeff(0, 0), 1);
    EXPECT_EQ(energy.hessian(Vec::Ones(1), HessianKind::Projected).coeff(0, 0), 3);
}

TEST(MeshEnergy, FeasibleStepUsesMatchingLocalDofValuesAndDeltas) {
    const auto topology = two_hexes();
    MeshEnergy energy(
        ElementDofMap(topology, 2),
        [](int, const Eigen::VectorXd& local_dof_values, Vec*, Eigen::MatrixXd*) {
            if (local_dof_values.minCoeff() <= 0)
                throw InfeasibleState("positive only");
            return local_dof_values.sum();
        },
        [](int, const Eigen::VectorXd& local_dof_values, const Eigen::VectorXd& local_dof_delta) {
            if (local_dof_values.minCoeff() <= 0)
                throw InfeasibleState("positive only");
            double alpha = 1;
            for (int j = 0; j < 2; ++j)
                if (local_dof_delta[j] < 0)
                    alpha = std::min(alpha, 0.9 * -local_dof_values[j] / local_dof_delta[j]);
            return alpha;
        });
    Vec x(4), dx(4);
    x << 1, 2, 3, 4;
    dx << -2, 1, -12, 0;
    EXPECT_DOUBLE_EQ(energy.max_feasible_step(x, dx), 0.225);
    EXPECT_NO_THROW(energy.value(x + energy.max_feasible_step(x, dx) * dx));
    x[2] = -1;
    EXPECT_THROW(energy.value(x), InfeasibleState);
    EXPECT_THROW(energy.max_feasible_step(x, dx), InfeasibleState);
}

TEST(MeshEnergy, RejectsMalformedOutputsAndHandlesIndependentRequests) {
    const ConstantDofMap map(TriangleTopology(3, {{0, 1, 2}}), 2);
    MeshEnergy value_only(map, [](int, const Eigen::VectorXd&, Vec* g, Eigen::MatrixXd* h) {
        if (g || h)
            throw std::logic_error("unsupported");
        return 2.;
    });
    EXPECT_EQ(value_only.value(Vec::Zero(2)), 2);
    EXPECT_THROW(value_only.gradient(Vec::Zero(2)), std::logic_error);
    EXPECT_THROW(value_only.hessian(Vec::Zero(2)), std::logic_error);
    MeshEnergy second_only(map, [](int, const Eigen::VectorXd&, Vec* g, Eigen::MatrixXd* h) {
        if (g)
            throw std::logic_error("unsupported");
        if (h)
            *h = Eigen::Matrix2d::Identity();
        return 0.;
    });
    EXPECT_TRUE(Eigen::MatrixXd(second_only.hessian(Vec::Zero(2))).isIdentity());
    MeshEnergy bad(map, [](int, const Eigen::VectorXd&, Vec* g, Eigen::MatrixXd* h) {
        if (g)
            *g = Vec::Ones(1);
        if (h)
            *h = Eigen::MatrixXd::Ones(1, 2);
        return 0.;
    });
    EXPECT_THROW(bad.gradient(Vec::Zero(2)), std::logic_error);
    EXPECT_THROW(bad.hessian(Vec::Zero(2)), std::logic_error);
    MeshEnergy unwritten(map, [](int, const Eigen::VectorXd&, Vec*, Eigen::MatrixXd*) { return 0.; });
    EXPECT_THROW(unwritten.gradient(Vec::Zero(2)), std::logic_error);
    EXPECT_THROW(unwritten.hessian(Vec::Zero(2)), std::logic_error);
    MeshEnergy bad_step(
        map, [](int, const Eigen::VectorXd&, Vec*, Eigen::MatrixXd*) { return 0.; },
        [](int, const Eigen::VectorXd&, const Eigen::VectorXd&) { return std::numeric_limits<double>::quiet_NaN(); });
    EXPECT_THROW(bad_step.max_feasible_step(Vec::Zero(2), Vec::Ones(2)), std::logic_error);
    EXPECT_THROW(bad.value(Vec::Zero(1)), std::invalid_argument);
    EXPECT_THROW(bad.value(Vec::Constant(2, std::numeric_limits<double>::infinity())), std::invalid_argument);
    EXPECT_THROW(
        (MeshEnergy(VertexDofMap(two_hexes(), 0), [](int, const auto&, Vec*, Eigen::MatrixXd*) { return 0.; })),
        std::invalid_argument);
}

TEST(MeshEnergy, EmptyMeshProducesZeroOutputs) {
    MeshEnergy energy(VertexDofMap(HexTopology(2, {}), 3), [](int, const auto&, Vec*, Eigen::MatrixXd*) -> double {
        throw std::logic_error("must not run");
    });
    EXPECT_EQ(energy.value(Vec::Zero(6)), 0);
    EXPECT_TRUE(energy.gradient(Vec::Zero(6)).isZero());
    EXPECT_EQ(energy.hessian(Vec::Zero(6)).nonZeros(), 0);
}

TEST(MeshEnergy, RejectsWrongSparsePatternBeforeCallbacksOrOutputWrites) {
    int calls = 0;
    const ElementDofMap map(TriangleTopology(3, {{0, 1, 2}}), 2);
    MeshEnergy energy(map, [&calls](int, const auto&, Vec*, Eigen::MatrixXd*) {
        ++calls;
        return 0.;
    });
    Vec gradient = Vec::Constant(1, 7);
    SpMat hessian(2, 2);
    hessian.insert(0, 0) = 9;
    hessian.makeCompressed();
    const SpMat before = hessian;
    EXPECT_THROW(energy.evaluate(Vec::Zero(2), &gradient, &hessian), std::invalid_argument);
    EXPECT_EQ(calls, 0);
    EXPECT_EQ(gradient, Vec::Constant(1, 7));
    EXPECT_EQ(Eigen::MatrixXd(hessian), Eigen::MatrixXd(before));
}

TEST(MeshEnergy, RejectsProjectionOverflowBeforeScatter) {
    const ConstantDofMap map(TriangleTopology(3, {{0, 1, 2}}), 2);
    MeshEnergy energy(map, [](int, const auto&, Vec*, Eigen::MatrixXd* h) {
        if (h)
            *h = Eigen::Matrix2d::Constant(std::numeric_limits<double>::max());
        return 0.;
    });
    const Vec x = Vec::Zero(2);
    EXPECT_TRUE(Eigen::MatrixXd(energy.hessian(x)).allFinite());
    EXPECT_THROW(energy.hessian(x, HessianKind::Projected), std::logic_error);
}
} // namespace
