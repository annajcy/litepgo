#include "../../fem/material/volumetric/test_material_sources.hpp"
#include "../../testing/batch_fixtures.hpp"
#include "../../testing/element_formula_energy.hpp"
#include "../../testing/evaluate_energy.hpp"
#include "litepgo/dof/constant_dof_map.hpp"
#include "litepgo/dof/element_dof_map.hpp"
#include "litepgo/dof/stencil_dof_map.hpp"
#include "litepgo/dof/vertex_dof_map.hpp"
#include "litepgo/energy/scalar/mesh_energy.hpp"
#include "litepgo/fem/basis/3d/hex_q1_basis.hpp"
#include "litepgo/fem/field/identity_field.hpp"
#include <future>

#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/material/volumetric/volumetric_material_local_energy.hpp"
#include "litepgo/fem/quadrature/3d/gauss_legendre_3d.hpp"
#include "litepgo/mesh/triangle_mid_edge_stencils.hpp"
#include <Eigen/LU>
#include <atomic>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <tbb/global_control.h>

namespace litepgo::energy::test {

namespace {
using namespace litepgo::fem::test;
using litepgo::testing::ElementFormulaEnergy;

mesh::HexTopology two_hexes() {
    return mesh::HexTopology(12, {{1, 3, 9, 4, 7, 6, 0, 10}, {3, 8, 5, 9, 6, 2, 11, 0}});
}
fem::InterpolatedField3D<fem::HexQ1Basis> make_X_field() {
    Eigen::Matrix<double, 8, 3> nodes;
    for (int i = 0; i < 8; ++i)
        nodes.row(i) = fem::HexQ1Basis::coefficient_info(i).point.transpose();
    return fem::InterpolatedField3D<fem::HexQ1Basis>(nodes);
}
auto passive_fields() {
    Eigen::Matrix3d F;
    F << 1.1, 0.04, 0.02, -0.03, 0.95, 0.01, 0.02, 0.03, 1.05;
    return MaterialSources{fem::IdentityField3D(F.reshaped()),
                           fem::IdentityField3D(Eigen::Matrix3d::Identity().reshaped()),
                           fem::IdentityField3D(Eigen::Vector2d(2, 3))};
}

void check_fd(ElementFormulaEnergy& energy, const Vec& x) {
    Vec gradient;
    SpMat hessian;
    const double value = litepgo::testing::evaluate_energy(energy, x, &gradient, &hessian);
    EXPECT_DOUBLE_EQ(value, energy.eval_value(x));
    EXPECT_TRUE(gradient.isApprox(energy.eval_gradient(x), 1e-13));
    EXPECT_TRUE(Eigen::MatrixXd(hessian).isApprox(Eigen::MatrixXd(energy.eval_hessian(x)), 1e-13));
    const double step = 1e-5;
    for (Eigen::Index j = 0; j < x.size(); ++j) {
        Vec plus = x, minus = x;
        plus[j] += step;
        minus[j] -= step;
        EXPECT_NEAR(gradient[j], (energy.eval_value(plus) - energy.eval_value(minus)) / (2 * step), 1e-7);
        const Vec fd = (energy.eval_gradient(plus) - energy.eval_gradient(minus)) / (2 * step);
        EXPECT_LT((fd - Eigen::MatrixXd(hessian).col(j)).norm(), 1e-6);
    }
    {
        tbb::global_control threads(tbb::global_control::max_allowed_parallelism, 1);
        Vec g;
        SpMat h;
        EXPECT_DOUBLE_EQ(litepgo::testing::evaluate_energy(energy, x, &g, &h), value);
        EXPECT_EQ(g, gradient);
        EXPECT_EQ(Eigen::MatrixXd(h), Eigen::MatrixXd(hessian));
    }
    {
        tbb::global_control threads(tbb::global_control::max_allowed_parallelism, 4);
        Vec g;
        SpMat h;
        EXPECT_DOUBLE_EQ(litepgo::testing::evaluate_energy(energy, x, &g, &h), value);
        EXPECT_EQ(g, gradient);
        EXPECT_EQ(Eigen::MatrixXd(h), Eigen::MatrixXd(hessian));
    }
    EXPECT_NO_THROW(energy.eval_hessian(x));
}

template <class Map, class Fields, class Bind>
void check_material_path(const Map& map, const Vec& data, Fields fields, Bind bind) {
    ElementFormulaEnergy energy(map, [fields, bind](int, const auto& x_local, Vec* g, Eigen::MatrixXd* h) {
        const auto X_field = make_X_field();
        const auto sources = fields(x_local);
        const auto local_energy = bind(sources);
        if (g)
            g->setZero(local_energy.num_dofs());
        if (h)
            h->setZero(local_energy.num_dofs(), local_energy.num_dofs());
        using Q = fem::GaussLegendre3D<2>;
        double result = 0;
        for (int q = 0; q < Q::num_points; ++q) {
            const Eigen::Vector3d xi = Q::points().row(q).transpose();
            const double weight = Q::weights()[q] * X_field.eval_dvalue_dxi(xi).determinant();
            result += local_energy.accumulate(xi, weight, g, h);
        }
        return result;
    });
    // This test only evaluates nearby feasible states; it does not use the default step for a solver.
    check_fd(energy, data);
}

TEST(MeshEnergy, HexNodeDisplacement) {
    const auto topology = two_hexes();
    const dof::VertexDofMap map(topology, 3);
    Vec data = Vec::Zero(36);
    for (int i = 0; i < data.size(); ++i)
        data[i] = 0.01 * std::sin(i + 0.3);
    check_material_path(
        map, data,
        [](const auto& x_local) {
            EXPECT_EQ(x_local.rows(), 8);
            EXPECT_EQ(x_local.cols(), 3);
            auto fields = passive_fields();
            return MaterialSources{
                OwnedDeformationField(fem::InterpolatedField3D<fem::HexQ1Basis>(x_local), make_X_field()), fields.Fp,
                fields.theta};
        },
        [](const auto& fields) { return bind_energy<fem::VolumetricMaterialLocalEnergyF3D<fem::NeoHookean>>(fields); });
}

TEST(MeshEnergy, OwnsPlanUsesCurrentXAndSupportsTriangleAndTet) {
    const auto quadratic = [](int e, const auto& x_local, Vec* g, Eigen::MatrixXd* h) {
        const Vec flat = x_local.template reshaped<Eigen::RowMajor>();
        if (g)
            *g = (e + 1) * flat;
        if (h)
            *h = (e + 1) * Eigen::MatrixXd::Identity(flat.size(), flat.size());
        return 0.5 * (e + 1) * flat.squaredNorm();
    };
    ElementFormulaEnergy triangle(dof::VertexDofMap(mesh::TriangleTopology(4, {{2, 0, 1}, {1, 0, 3}}), 2), quadratic);
    Vec x = Vec::LinSpaced(8, 0.1, 0.8);
    check_fd(triangle, x);
    Vec counts(8);
    counts << 3, 3, 3, 3, 1, 1, 2, 2;
    EXPECT_TRUE(triangle.eval_gradient(x).isApprox(counts.cwiseProduct(x), 1e-14));
    EXPECT_TRUE(Eigen::MatrixXd(triangle.eval_hessian(x)).isApprox(Eigen::MatrixXd(counts.asDiagonal()), 1e-14));
    EXPECT_NE(triangle.eval_value(x), triangle.eval_value(2 * x));
    EXPECT_DOUBLE_EQ(triangle.max_feasible_step(x, x), 1);
    ElementFormulaEnergy tet(dof::VertexDofMap(mesh::TetTopology(4, {{3, 1, 2, 0}}), 1), quadratic);
    check_fd(tet, Vec::LinSpaced(4, 0.1, 0.4));
}

TEST(MeshEnergy, StencilMapsAssembleNeighborCouplingsAndVariableLocalSizes) {
    const mesh::TriangleTopology topology(10, {{4, 1, 5}, {5, 1, 3}, {4, 5, 0}, {1, 4, 2}, {6, 7, 8}});
    const auto stencils = mesh::make_triangle_mid_edge_stencils(topology);
    auto energy = [&] {
        const dof::StencilDofMap map(topology.num_vertices(), stencils.vertices, 3);
        return ElementFormulaEnergy(map, [](int e, const Eigen::MatrixXd& local, Vec* g, Eigen::MatrixXd* h) {
            const Vec q = local.reshaped<Eigen::RowMajor>();
            if (g)
                *g = Vec::Constant(q.size(), (e + 1) * q.sum()) + q.array().cube().matrix();
            if (h) {
                h->setConstant(q.size(), q.size(), e + 1);
                h->diagonal().array() += 3 * q.array().square();
            }
            return 0.5 * (e + 1) * q.sum() * q.sum() + 0.25 * q.array().pow(4).sum();
        });
    }();
    const Vec x = Vec::LinSpaced(energy.num_dofs(), 0.1, 0.9);
    check_fd(energy, x);

    Vec expected_g = Vec::Zero(x.size());
    Eigen::MatrixXd expected_h = Eigen::MatrixXd::Zero(x.size(), x.size());
    double expected_value = 0;
    for (int e = 0; e < topology.num_cells(); ++e) {
        const auto& vertices = stencils.vertices[static_cast<std::size_t>(e)];
        double sum = 0, fourth_powers = 0;
        for (int v : vertices)
            for (int c = 0; c < 3; ++c) {
                sum += x[3 * v + c];
                fourth_powers += std::pow(x[3 * v + c], 4);
            }
        expected_value += 0.5 * (e + 1) * sum * sum + 0.25 * fourth_powers;
        for (int v : vertices)
            for (int c = 0; c < 3; ++c) {
                const int i = 3 * v + c;
                expected_g[i] += (e + 1) * sum + std::pow(x[i], 3);
                expected_h(i, i) += 3 * x[i] * x[i];
                for (int w : vertices)
                    for (int d = 0; d < 3; ++d)
                        expected_h(i, 3 * w + d) += e + 1;
            }
    }
    EXPECT_NEAR(energy.eval_value(x), expected_value, 1e-12);
    EXPECT_TRUE(energy.eval_gradient(x).isApprox(expected_g, 1e-13));
    const Eigen::MatrixXd actual_h = energy.eval_hessian(x);
    EXPECT_TRUE(actual_h.isApprox(expected_h, 1e-13));
    // Vertices 3 and 0 share no triangle, but both support the central face's stencil.
    EXPECT_DOUBLE_EQ(actual_h(3 * 3, 3 * 0 + 1), 1);
    EXPECT_TRUE(actual_h.bottomRows(3).isZero());
    EXPECT_TRUE(energy.eval_gradient(x).tail(3).isZero());
    EXPECT_DOUBLE_EQ(energy.max_feasible_step(x, x), 1);
}

TEST(MeshEnergy, ProjectsEachLocalHessianBeforeScatter) {
    const auto topology = two_hexes();
    ElementFormulaEnergy energy(dof::ConstantDofMap(topology, 1),
                                [](int e, const Eigen::VectorXd& x_local, Vec* g, Eigen::MatrixXd* h) {
                                    double k = e == 0 ? -2 : 3;
                                    if (g)
                                        *g = k * x_local;
                                    if (h)
                                        *h = Eigen::MatrixXd::Constant(1, 1, k);
                                    return 0.5 * k * x_local.squaredNorm();
                                });
    EXPECT_EQ(energy.eval_hessian(Vec::Ones(1)).coeff(0, 0), 1);
    EXPECT_EQ(energy.eval_hessian(Vec::Ones(1), HessianKind::Projected).coeff(0, 0), 3);
}

TEST(MeshEnergy, FeasibleStepUsesMatchingLocalDofValuesAndDeltas) {
    const auto topology = two_hexes();
    ElementFormulaEnergy energy(
        dof::ElementDofMap(topology, 2),
        [](int, const Eigen::VectorXd& x_local, Vec*, Eigen::MatrixXd*) {
            if (x_local.minCoeff() <= 0)
                throw std::domain_error("positive only");
            return x_local.sum();
        },
        [](int, const Eigen::VectorXd& x_local, const Eigen::VectorXd& dx_local) {
            if (x_local.minCoeff() <= 0)
                throw std::domain_error("positive only");
            double alpha = 1;
            for (int j = 0; j < 2; ++j)
                if (dx_local[j] < 0)
                    alpha = std::min(alpha, 0.9 * -x_local[j] / dx_local[j]);
            return alpha;
        });
    Vec x(4), dx(4);
    x << 1, 2, 3, 4;
    dx << -2, 1, -12, 0;
    EXPECT_DOUBLE_EQ(energy.max_feasible_step(x, dx), 0.225);
    EXPECT_NO_THROW(energy.eval_value(x + energy.max_feasible_step(x, dx) * dx));
    x[2] = -1;
    EXPECT_THROW(energy.eval_value(x), std::domain_error);
    EXPECT_THROW(energy.max_feasible_step(x, dx), std::domain_error);
}

TEST(MeshEnergy, RejectsMalformedOutputsAndHandlesIndependentRequests) {
    const dof::ConstantDofMap map(mesh::TriangleTopology(3, {{0, 1, 2}}), 2);
    ElementFormulaEnergy value_only(map, [](int, const Eigen::VectorXd&, Vec* g, Eigen::MatrixXd* h) {
        if (g || h)
            throw std::logic_error("unsupported");
        return 2.;
    });
    EXPECT_EQ(value_only.eval_value(Vec::Zero(2)), 2);
    EXPECT_THROW(value_only.eval_gradient(Vec::Zero(2)), std::logic_error);
    EXPECT_THROW(value_only.eval_hessian(Vec::Zero(2)), std::logic_error);
    ElementFormulaEnergy second_only(map, [](int, const Eigen::VectorXd&, Vec* g, Eigen::MatrixXd* h) {
        if (g)
            throw std::logic_error("unsupported");
        if (h)
            *h = Eigen::Matrix2d::Identity();
        return 0.;
    });
    EXPECT_TRUE(Eigen::MatrixXd(second_only.eval_hessian(Vec::Zero(2))).isIdentity());
    ElementFormulaEnergy bad(map, [](int, const Eigen::VectorXd&, Vec* g, Eigen::MatrixXd* h) {
        if (g)
            *g = Vec::Ones(1);
        if (h)
            *h = Eigen::MatrixXd::Ones(1, 2);
        return 0.;
    });
    EXPECT_THROW(bad.eval_gradient(Vec::Zero(2)), std::logic_error);
    EXPECT_THROW(bad.eval_hessian(Vec::Zero(2)), std::logic_error);
    ElementFormulaEnergy unwritten(map, [](int, const Eigen::VectorXd&, Vec*, Eigen::MatrixXd*) { return 0.; });
    EXPECT_THROW(unwritten.eval_gradient(Vec::Zero(2)), std::logic_error);
    EXPECT_THROW(unwritten.eval_hessian(Vec::Zero(2)), std::logic_error);
    ElementFormulaEnergy bad_step(
        map, [](int, const Eigen::VectorXd&, Vec*, Eigen::MatrixXd*) { return 0.; },
        [](int, const Eigen::VectorXd&, const Eigen::VectorXd&) { return std::numeric_limits<double>::quiet_NaN(); });
    EXPECT_THROW(bad_step.max_feasible_step(Vec::Zero(2), Vec::Ones(2)), std::logic_error);
    EXPECT_THROW(bad.eval_value(Vec::Zero(1)), std::invalid_argument);
    EXPECT_THROW(bad.eval_value(Vec::Constant(2, std::numeric_limits<double>::infinity())), std::invalid_argument);
    EXPECT_THROW((ElementFormulaEnergy(dof::VertexDofMap(two_hexes(), 0),
                                       [](int, const auto&, Vec*, Eigen::MatrixXd*) { return 0.; })),
                 std::invalid_argument);
}

TEST(MeshEnergy, EmptyMeshProducesZeroOutputs) {
    ElementFormulaEnergy energy(
        dof::VertexDofMap(mesh::HexTopology(2, {}), 3),
        [](int, const auto&, Vec*, Eigen::MatrixXd*) -> double { throw std::logic_error("must not run"); });
    EXPECT_EQ(energy.eval_value(Vec::Zero(6)), 0);
    EXPECT_TRUE(energy.eval_gradient(Vec::Zero(6)).isZero());
    EXPECT_EQ(energy.eval_hessian(Vec::Zero(6)).nonZeros(), 0);
    EXPECT_DOUBLE_EQ(energy.max_feasible_step(Vec::Zero(6), Vec::Ones(6)), 1);
}

TEST(MeshEnergy, RejectsWrongInputBeforeCallbacks) {
    int calls = 0;
    const dof::ElementDofMap map(mesh::TriangleTopology(3, {{0, 1, 2}}), 2);
    ElementFormulaEnergy energy(map, [&calls](int, const auto&, Vec*, Eigen::MatrixXd*) {
        ++calls;
        return 0.;
    });
    Vec gradient = Vec::Constant(1, 7);
    SpMat hessian(2, 2);
    hessian.insert(0, 0) = 9;
    hessian.makeCompressed();
    const SpMat before = hessian;
    EXPECT_THROW(litepgo::testing::evaluate_energy(energy, Vec::Zero(3), &gradient, &hessian), std::invalid_argument);
    EXPECT_EQ(calls, 0);
    EXPECT_EQ(gradient, Vec::Constant(1, 7));
    EXPECT_EQ(Eigen::MatrixXd(hessian), Eigen::MatrixXd(before));
}

TEST(MeshEnergy, RejectsProjectionOverflowAtAssemblyBoundary) {
    const dof::ConstantDofMap map(mesh::TriangleTopology(3, {{0, 1, 2}}), 2);
    ElementFormulaEnergy energy(map, [](int, const auto&, Vec*, Eigen::MatrixXd* h) {
        if (h)
            *h = Eigen::Matrix2d::Constant(std::numeric_limits<double>::max());
        return 0.;
    });
    const Vec x = Vec::Zero(2);
    EXPECT_TRUE(Eigen::MatrixXd(energy.eval_hessian(x)).allFinite());
    EXPECT_THROW(energy.eval_hessian(x, HessianKind::Projected), std::domain_error);
}

TEST(MeshEnergy, OverflowingSumsOfFiniteElementValuesAndDerivativesThrowDomainError) {
    const dof::ConstantDofMap map(mesh::TriangleTopology(3, {{0, 1, 2}, {0, 1, 2}}), 2);
    const Vec x = Vec::Zero(2);
    for (int request : {0, 1, 2}) {
        double scale = std::numeric_limits<double>::max();
        ElementFormulaEnergy energy(map, [&](int, const Vec&, Vec* g, Eigen::MatrixXd* H) {
            if (g)
                g->setConstant(2, request == 1 ? scale : 0);
            if (H)
                H->setConstant(2, 2, request == 2 ? scale : 0);
            return request == 0 ? scale : 0;
        });
        if (request == 0)
            EXPECT_THROW(energy.eval_value(x), std::domain_error);
        if (request == 1)
            EXPECT_THROW(energy.eval_gradient(x), std::domain_error);
        if (request == 2)
            EXPECT_THROW(energy.eval_hessian(x), std::domain_error);
        scale = 1;
        EXPECT_TRUE(std::isfinite(energy.eval_value(x)));
        EXPECT_TRUE(energy.eval_gradient(x).allFinite());
        EXPECT_TRUE(Eigen::MatrixXd(energy.eval_hessian(x)).allFinite());
    }
}

struct SharedDofMap {
    std::vector<std::vector<int>> indices;
    SharedDofMap() {
        for (int e = 0; e < 513; ++e)
            indices.push_back(e % 13 == 0 ? std::vector<int>{}
                                          : std::vector<int>{e % 256, (e + 1) % 256, (7 * e) % 256, e % 256});
    }
    int num_dofs() const {
        return 257; // The last DOF is unused.
    }
    int num_elements() const {
        return static_cast<int>(indices.size());
    }
    std::vector<int> dof_indices(int e) const {
        return indices[static_cast<std::size_t>(e)];
    }
    Vec gather(int e, const ConstVecRef& x) const {
        const auto& dofs = indices[static_cast<std::size_t>(e)];
        Vec local(dofs.size());
        for (std::size_t j = 0; j < dofs.size(); ++j)
            local[static_cast<Eigen::Index>(j)] = x[dofs[j]];
        return local;
    }
};

TEST(MeshEnergy, ParallelAssemblyMatchesSerialWithSharedRepeatedAndUnusedDofs) {
    const SharedDofMap map;
    const auto local = [](int e, const Vec& x_local, Vec* g, Eigen::MatrixXd* h) {
        Eigen::MatrixXd A = Eigen::MatrixXd::Constant(x_local.size(), x_local.size(), 0.01 * (e % 3));
        A.diagonal().array() += 1 + e % 5;
        const Vec derivative = A * x_local;
        if (g)
            *g = derivative;
        if (h)
            *h = A;
        return 0.5 * x_local.dot(derivative);
    };
    ElementFormulaEnergy energy(map, local);
    const dof::EnergyAssembler serial(map.num_dofs(), map.indices);
    for (int state = 0; state < 3; ++state) {
        const Vec x = Vec::LinSpaced(map.num_dofs(), -0.4 + state, 0.7 + state);
        Vec expected_g = Vec::Zero(map.num_dofs());
        SpMat expected_h = serial.hessian_pattern();
        double expected_value = 0;
        for (int e = 0; e < map.num_elements(); ++e) {
            Vec g;
            Eigen::MatrixXd h;
            expected_value += local(e, map.gather(e, x), &g, &h);
            serial.add_gradient(e, g, expected_g);
            serial.add_hessian(e, h, expected_h);
        }
        for (int threads : {1, 2, 4, 8}) {
            tbb::global_control limit(tbb::global_control::max_allowed_parallelism, threads);
            for (int request = 0; request < 4; ++request) {
                SCOPED_TRACE(::testing::Message()
                             << "state=" << state << " threads=" << threads << " request=" << request);
                Vec g;
                SpMat h;
                EXPECT_EQ(litepgo::testing::evaluate_energy(energy, x, request & 1 ? &g : nullptr,
                                                            request & 2 ? &h : nullptr),
                          expected_value);
                if (request & 1)
                    EXPECT_EQ(g, expected_g);
                if (request & 2)
                    EXPECT_EQ(Eigen::MatrixXd(h), Eigen::MatrixXd(expected_h));
            }
        }
    }
}

TEST(MeshEnergy, ReusedBuffersRejectIncompleteWritesAndRecoverAfterExceptions) {
    bool write = true;
    ElementFormulaEnergy energy(dof::ConstantDofMap(mesh::TriangleTopology(3, {{0, 1, 2}, {0, 1, 2}}), 2),
                                [&write](int e, const Vec& x_local, Vec* g, Eigen::MatrixXd* h) {
                                    if (write && g)
                                        *g = 2 * (e + 1) * x_local;
                                    if (write && h)
                                        *h = 2 * (e + 1) * Eigen::Matrix2d::Identity();
                                    return (e + 1) * x_local.squaredNorm();
                                });
    Vec x = Eigen::Vector2d(0.2, 0.4), g;
    SpMat h;
    for (int request : {1, 2, 3}) {
        EXPECT_DOUBLE_EQ(
            litepgo::testing::evaluate_energy(energy, x, request & 1 ? &g : nullptr, request & 2 ? &h : nullptr),
            3 * x.squaredNorm());
        write = false;
        EXPECT_THROW(
            litepgo::testing::evaluate_energy(energy, x, request & 1 ? &g : nullptr, request & 2 ? &h : nullptr),
            std::logic_error);
        write = true;
        x.array() += 0.1;
        EXPECT_DOUBLE_EQ(litepgo::testing::evaluate_energy(energy, x, &g, &h), 3 * x.squaredNorm());
        EXPECT_EQ(g, 6 * x);
        EXPECT_EQ(Eigen::MatrixXd(h), 6 * Eigen::Matrix2d::Identity());
    }
}

TEST(MeshEnergy, ParallelFeasibleStepMatchesSerialAndPropagatesErrors) {
    const SharedDofMap map;
    std::atomic<int> calls = 0;
    bool bad_bound = false;
    const auto step = [&calls, &bad_bound](int e, const Vec& x_local, const Vec& dx_local) {
        ++calls;
        if (bad_bound && e == 500)
            return std::numeric_limits<double>::quiet_NaN();
        double alpha = 1;
        for (Eigen::Index j = 0; j < x_local.size(); ++j) {
            if (x_local[j] <= 0)
                throw std::domain_error("positive DOFs required");
            if (dx_local[j] < 0)
                alpha = std::min(alpha, 0.8 * x_local[j] / -dx_local[j]);
        }
        return alpha;
    };
    ElementFormulaEnergy energy(map, [](int, const Vec&, Vec*, Eigen::MatrixXd*) { return 0.; }, step);
    Vec x = Vec::LinSpaced(map.num_dofs(), 1, 2);
    const Vec dx = -Vec::LinSpaced(map.num_dofs(), 0.1, 3);
    double expected = 1;
    for (int e = 0; e < map.num_elements(); ++e)
        expected = std::min(expected, step(e, map.gather(e, x), map.gather(e, dx)));
    for (int threads : {1, 2, 4, 8}) {
        tbb::global_control limit(tbb::global_control::max_allowed_parallelism, threads);
        calls = 0;
        EXPECT_DOUBLE_EQ(energy.max_feasible_step(x, dx), expected);
        EXPECT_EQ(calls, map.num_elements());
        bad_bound = true;
        EXPECT_THROW(energy.max_feasible_step(x, dx), std::logic_error);
        bad_bound = false;
        x[0] = -1;
        EXPECT_THROW(energy.max_feasible_step(x, dx), std::domain_error);
        x[0] = 1;
        EXPECT_DOUBLE_EQ(energy.max_feasible_step(x, dx), expected);
    }
}
} // namespace

} // namespace litepgo::energy::test

#include "litepgo/energy/scalar/mesh_energy.hpp"

TEST(MeshEnergy, GroupsVariableStencilsAndAssembles) {
    using namespace litepgo;
    dof::StencilDofMap map(3, {{0, 1}, {1}, {1, 2}}, 3);
    ::litepgo::testing::BatchEnergyFixture batched(
        map,
        [](std::span<const int> ids, const energy::BatchMatrix& x, energy::BatchMatrix* g, energy::BatchMatrix* h) {
            EXPECT_LE(ids.size(), 2u);
            if (g)
                *g = x;
            if (h) {
                h->setZero(x.rows() * x.cols(), x.cols());
                for (Eigen::Index b = 0; b < x.rows(); ++b)
                    h->middleRows(b * x.cols(), x.cols()).setIdentity();
            }
            return Vec(0.5 * x.rowwise().squaredNorm());
        },
        2);
    const Vec x = Vec::Ones(9);
    Vec expected(9);
    expected << 1, 1, 1, 3, 3, 3, 1, 1, 1;
    EXPECT_DOUBLE_EQ(batched.eval_value(x), 7.5);
    EXPECT_TRUE(batched.eval_gradient(x).isApprox(expected));
    EXPECT_TRUE(Eigen::MatrixXd(batched.eval_hessian(x)).isApprox(Eigen::MatrixXd(expected.asDiagonal())));
    EXPECT_THROW((::litepgo::testing::BatchEnergyFixture(map, {}, 0)), std::invalid_argument);
}

#include "litepgo/energy/response/mesh_response.hpp"
#include <chrono>
#include <tbb/task_arena.h>
#include <thread>

TEST(MeshEnergy, BatchCallbacksRunConcurrentlyForEnergyResponseAndStep) {
    using namespace litepgo;
    tbb::global_control limit(tbb::global_control::max_allowed_parallelism, 4);
    tbb::task_arena arena(4);
    std::vector<std::vector<int>> indices(64, {0});
    dof::StencilDofMap map(1, indices, 1);
    std::atomic<int> active{0}, peak{0}, calls{0};
    const auto visit = [&] {
        const int count = ++active;
        int previous = peak.load();
        while (previous < count && !peak.compare_exchange_weak(previous, count)) {
        }
        ++calls;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        --active;
    };
    ::litepgo::testing::BatchEnergyFixture E(
        map,
        [&](auto, const energy::BatchMatrix& x, auto* g, auto* H) {
            visit();
            if (g)
                *g = x;
            if (H)
                H->setOnes(x.rows(), 1);
            return Vec(0.5 * x.rowwise().squaredNorm());
        },
        1,
        [&](auto, const auto& x, const auto&) {
            visit();
            return Vec::Constant(x.rows(), 0.5).eval();
        });
    ::litepgo::testing::BatchResponseFixture R(
        map, map,
        [&](auto, const energy::BatchMatrix& x, auto* J) {
            visit();
            if (J)
                J->setOnes(x.rows(), 1);
            return x;
        },
        1);
    const Vec x = Vec::Ones(1);
    const auto check = [&](auto operation) {
        calls = 0;
        peak = 0;
        arena.execute(operation);
        EXPECT_EQ(calls, 64);
        EXPECT_GT(peak, 1);
        EXPECT_EQ(active, 0);
    };
    check([&] { EXPECT_DOUBLE_EQ(E.eval_value(x), 32); });
    check([&] { EXPECT_DOUBLE_EQ(E.max_feasible_step(x, x), 0.5); });
    check([&] { EXPECT_DOUBLE_EQ(R.eval_value(x)[0], 64); });
}

TEST(MeshEnergy, EmptyLocalDofsStillHaveOneValuePerElement) {
    using namespace litepgo;
    const dof::StencilDofMap map(1, {{}, {}}, 1);
    ::litepgo::testing::BatchEnergyFixture E(
        map,
        [](auto ids, const auto&, auto* g, auto* H) {
            if (g)
                g->resize(ids.size(), 0);
            if (H)
                H->resize(0, 0);
            return Vec::Ones(ids.size()).eval();
        },
        1);
    const Vec x = Vec::Zero(1);
    EXPECT_DOUBLE_EQ(E.eval_value(x), 2);
    EXPECT_TRUE(E.eval_gradient(x).isZero());
    EXPECT_EQ(E.eval_hessian(x).nonZeros(), 0);
    EXPECT_EQ(E.eval_hessian(x, energy::HessianKind::Projected).nonZeros(), 0);
}

TEST(MeshEnergy, SharedWorkspaceRejectsOverlappingStepAndRecovers) {
    using namespace litepgo;
    std::promise<void> entered, release;
    auto entered_future = entered.get_future();
    auto release_future = release.get_future().share();
    const dof::ElementDofMap map(mesh::TriangleTopology(3, {{0, 1, 2}}), 2);
    bool wait = true;
    litepgo::testing::ElementFormulaEnergy energy(map, [&](int, const auto&, Vec* g, Eigen::MatrixXd* H) {
        if (wait) {
            entered.set_value();
            release_future.wait();
        }
        if (g)
            g->setZero();
        if (H)
            H->setZero();
        return 0.;
    });
    const Vec x = Vec::Zero(2);
    auto evaluation = std::async(std::launch::async, [&] { return energy.eval_value(x); });
    entered_future.wait();
    EXPECT_THROW(energy.max_feasible_step(x, x), std::logic_error);
    release.set_value();
    EXPECT_EQ(evaluation.get(), 0);
    wait = false;
    EXPECT_EQ(energy.eval_value(x), 0);
}
