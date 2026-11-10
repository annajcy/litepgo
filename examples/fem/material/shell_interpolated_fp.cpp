// Native counterpart of shell_interpolated_fp.py: no fixed DOFs, PBC or contact.
#include "litepgo/fem/assembly/basis_dof_transform.hpp"
#include "litepgo/fem/assembly/shell_fem_energy.hpp"
#include "litepgo/fem/basis/3d/hex_hermite_basis.hpp"
#include "litepgo/fem/basis/3d/hex_q1_basis.hpp"
#include "litepgo/fem/material/shell/koiter_stvk.hpp"
#include "litepgo/fem/material/shell/pointwise_shell_adapter.hpp"
#include "litepgo/fem/response/deformation_gradient_response.hpp"
#include "litepgo/fem/response/interpolated_response.hpp"
#include "litepgo/linalg/sparse_solver/auto_sparse_solver.hpp"
#include "litepgo/mesh/io/obj.hpp"
#include "litepgo/mesh/triangle_mid_edge_stencils.hpp"
#include "litepgo/solver/newton.hpp"
#include <fstream>
#include <iostream>
#include <numbers>
#include <tbb/global_control.h>
using namespace litepgo;

template <class Basis> void run(const mesh::TriangleMesh& input, const std::filesystem::path& out, const char* name) {
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> X(input.positions.data(),
                                                                                  input.topology.num_vertices(), 3);
    Eigen::Vector3d lo = X.colwise().minCoeff(), hi = X.colwise().maxCoeff(), size = hi - lo, center = (lo + hi) / 2;
    Eigen::MatrixXd corners(8, 3);
    for (int i = 0; i < 8; ++i)
        corners.row(i) = (lo + size.cwiseProduct(fem::HexQ1Basis::coefficient_info(i).point)).transpose();
    auto geometry = std::make_shared<fem::InterpolatedResponse<fem::HexQ1Basis>>(corners);
    auto make_field = [&](bool affine) {
        Eigen::MatrixXd physical = Eigen::MatrixXd::Zero(8, 3 * Basis::num_basis_functions / 8);
        const double rate = std::numbers::pi / 3 / size.z();
        for (int i = 0; i < Basis::num_basis_functions; ++i) {
            const auto info = Basis::coefficient_info(i);
            const auto d = info.derivative_order;
            Eigen::Vector3d x = lo + size.cwiseProduct(info.point), value = Eigen::Vector3d::Zero();
            if (affine) {
                if (d[0] + d[1] + d[2] == 0)
                    value.y() = -.5 * (x.y() - center.y());
                else if (d[0] == 0 && d[1] == 1 && d[2] == 0)
                    value.y() = -.5;
            } else if (!(d[0] && d[1])) {
                double angle = rate * (x.z() - center.z()), c = std::cos(angle), s = std::sin(angle);
                Eigen::Matrix2d R, J;
                R << c, -s, s, c;
                J << 0, -1, 1, 0;
                Eigen::Vector2d v = (x - center).head<2>();
                if (d[0])
                    v = Eigen::Vector2d(1, 0);
                else if (d[1])
                    v = Eigen::Vector2d(0, 1);
                if (d[2])
                    value.head<2>() = rate * J * R * v;
                else
                    value.head<2>() = (R - Eigen::Matrix2d::Identity()) * v;
            }
            constexpr int slots = Basis::num_basis_functions / 8;
            physical.block<1, 3>(i / slots, 3 * (i % slots)) = value.transpose();
        }
        fem::BasisDofTransform<Basis> transform(*geometry);
        return std::make_shared<fem::InterpolatedResponse<Basis>>(transform.to_coefficients(physical));
    };
    auto up = make_field(false);
    fem::DeformationGradientResponse field(up, geometry), affine(make_field(true), geometry);
    for (const Eigen::Vector3d xi : {Eigen::Vector3d(.2, .3, .4), Eigen::Vector3d(.8, .6, .7)}) {
        auto av = affine.evaluate(xi)->value();
        Eigen::Matrix3d expected = Eigen::Vector3d(1, .5, 1).asDiagonal();
        if ((Eigen::Map<const Eigen::Matrix3d>(av.data()) - expected).norm() > 1e-12)
            throw std::runtime_error("Affine check failed");
        Eigen::Matrix3d fd;
        double h = 1e-6;
        for (int j = 0; j < 3; ++j)
            fd.col(j) = (up->evaluate(xi + h * Eigen::Vector3d::Unit(j))->value() -
                         up->evaluate(xi - h * Eigen::Vector3d::Unit(j))->value()) /
                        (2 * h * size[j]);
        auto fv = field.evaluate(xi)->value();
        if ((Eigen::Map<const Eigen::Matrix3d>(fv.data()) - Eigen::Matrix3d::Identity() - fd).norm() > 1e-8)
            throw std::runtime_error("Gradient check failed");
    }
    auto stencils = mesh::make_triangle_mid_edge_stencils(input.topology);
    dof::StencilDofMap map(input.topology.num_vertices(), stencils.vertices, 3);
    std::vector<fem::TriangleMidEdgeRestData> rest;
    std::vector<Eigen::Matrix3d> Fp;
    std::ofstream samples(out / (std::string(name) + "_Fp.csv"));
    samples.precision(17);
    for (int f = 0; f < map.num_elements(); ++f) {
        auto local = map.gather(f, input.positions);
        rest.emplace_back(local, stencils.opposite[f]);
        Eigen::Vector3d centroid = local.topRows(3).colwise().mean();
        Vec v = field.evaluate((centroid - lo).cwiseQuotient(size))->value();
        Fp.emplace_back(Eigen::Map<const Eigen::Matrix3d>(v.data()));
        if (!Fp.back().allFinite() || Fp.back().determinant() <= 0)
            throw std::runtime_error("Invalid Fp");
        for (int j = 0; j < 9; ++j)
            samples << v[j] << (j == 8 ? '\n' : ',');
    }
    Vec theta(2);
    theta << 1e8 * .35 / (1 - .35 * .35), 1e8 / (2 * 1.35);
    fem::ShellFemEnergy shell(std::make_shared<fem::PointwiseShellAdapter<fem::KoiterStVK>>(), map, rest,
                              std::vector<double>(rest.size(), .06), std::vector<Vec>(rest.size(), theta), Fp);
    solver::NewtonOptions options;
    options.max_iterations = 1500;
    options.project_hessian = false;
    options.max_coordinate_step = .01;
    linalg::AutoSparseSolver linear;
    auto result = solver::newton_solve(shell, dof::DofReduction(shell.num_dofs(), {}), Vec::Zero(shell.num_dofs()),
                                       options, linear, solver::AbsoluteConvergence(1e-5));
    double residual = shell.evaluate(result.x)->gradient().lpNorm<Eigen::Infinity>();
    if (result.status != solver::NewtonStatus::Converged || residual >= 1e-5)
        throw std::runtime_error("Relaxation failed");
    mesh::write_obj(out / (std::string(name) + "_relaxed.obj"),
                    mesh::TriangleMesh{input.positions + result.x, input.topology});
    std::ofstream stats(out / (std::string(name) + "_summary.txt"));
    stats.precision(17);
    stats << result.iterations << " " << result.energy << " " << residual << "\n";
    std::cout << name << ": iterations=" << result.iterations << ", E=" << result.energy << ", residual=" << residual
              << std::endl;
}
int main() {
    tbb::global_control threads(tbb::global_control::max_allowed_parallelism, 4);
    auto input = mesh::read_obj(LITEPGO_ASSETS_DIR "/mesh/schwarz_p_open_cell.obj");
    std::filesystem::path out = LITEPGO_OUTPUT_DIR "/shell_interpolated_fp_cpp";
    std::filesystem::create_directories(out);
    mesh::write_obj(out / "original.obj", input);
    run<fem::HexQ1Basis>(input, out, "q1");
    run<fem::HexHermiteBasis>(input, out, "hermite");
}
