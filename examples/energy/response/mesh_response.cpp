#include "litepgo/energy/response/affine_response.hpp"
#include "litepgo/fem/material/volumetric/pointwise_volumetric_adapter.hpp"
#include "litepgo/fem/response/interpolated_response.hpp"
#include <memory>
// Fit shared Neo-Hookean parameters from forces at a prescribed deformation, using batches.
// Build: cmake --build build/release --target mesh_response
// Run: build/release/examples/energy/response/mesh_response

#include "litepgo/dof/constant_dof_map.hpp"
#include "litepgo/dof/vertex_dof_map.hpp"
#include "litepgo/energy/response/mesh_response.hpp"
#include "litepgo/fem/assembly/volumetric_fem_energy.hpp"
#include "litepgo/fem/assembly/volumetric_fem_force_material_response.hpp"
#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/quadrature/3d/tet_centroid_quadrature.hpp"
#include "litepgo/solver/gauss_newton.hpp"
#include <Eigen/LU>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <tbb/global_control.h>

using namespace litepgo;

using Force = fem::VolumetricFemForceMaterialResponse<fem::TetP1Basis, fem::TetCentroidQuadrature>;

class ForceResidual final : public energy::FixedPatternResponse {
  public:
    ForceResidual(energy::Response& force, Vec observed)
        : FixedPatternResponse(Eigen::MatrixXd::Ones(force.num_outputs(), force.num_inputs()).sparseView()),
          force_(force), observed_(std::move(observed)) {}

  private:
    Vec eval_impl(const ConstVecRef& theta, SpMat* J) override {
        return (J ? ([&] {
                   auto evaluation = force_.evaluate(theta);
                   const auto derivative = evaluation->jacobian();
                   const Vec value = evaluation->value();
                   for (int c = 0; c < J->outerSize(); ++c)
                       for (SpMat::InnerIterator it(*J, c); it; ++it)
                           it.valueRef() = derivative.coeff(it.row(), c);
                   return value;
               }())
                  : force_.evaluate(theta)->value()) -
               observed_;
    }
    energy::Response& force_;
    Vec observed_;
};

int main(int argc, char** argv) {
    const int copies = argc > 1 ? std::stoi(argv[1]) : 1;
    const int batch_size = argc > 2 ? std::stoi(argv[2]) : 256;
    const int repeats = argc > 3 ? std::stoi(argv[3]) : 0;
    const int threads =
        argc > 4 ? std::stoi(argv[4])
                 : static_cast<int>(tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism));
    if (threads <= 0)
        throw std::invalid_argument("Thread count must be positive");
    tbb::global_control thread_limit(tbb::global_control::max_allowed_parallelism, threads);
    if (copies < 1 || batch_size < 1 || repeats < 0 || copies > 1000000)
        throw std::invalid_argument("Expected copies in [1,1000000], positive batch size, nonnegative repeats");
    Eigen::Matrix<double, 5, 3> X;
    X << 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1;
    Eigen::Matrix3d F;
    F << 1.15, .12, 0, .03, .94, .08, 0, .02, 1.07;
    const std::vector<mesh::TetTopology::Cell> base{{0, 1, 2, 3}, {1, 2, 3, 4}};
    std::vector<mesh::TetTopology::Cell> cells;
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> nodes(5 * copies, 3);
    for (int copy = 0; copy < copies; ++copy) {
        nodes.middleRows(5 * copy, 5) = X;
        for (auto cell : base) {
            for (int& vertex : cell)
                vertex += 5 * copy;
            cells.push_back(cell);
        }
    }
    const mesh::TetTopology topology(5 * copies, std::move(cells));
    const dof::VertexDofMap outputs(topology, 3);
    const dof::ConstantDofMap inputs(topology, 2);
    const Vec positions = nodes.reshaped<Eigen::RowMajor>();
    const Eigen::MatrixXd u = nodes * (F - Eigen::Matrix3d::Identity()).transpose();
    const Vec displacement = u.reshaped<Eigen::RowMajor>();
    std::vector<std::shared_ptr<litepgo::energy::Response>> geometry;
    for (int e = 0; e < topology.num_cells(); ++e)
        geometry.push_back(std::make_shared<fem::InterpolatedResponse<fem::TetP1Basis>>(outputs.gather(e, positions)));
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    const std::vector<std::shared_ptr<litepgo::energy::Response>> plastic(
        topology.num_cells(),
        std::make_shared<litepgo::energy::AffineResponse>(litepgo::SpMat((I.reshaped()).size(), 3), I.reshaped()));
    Force single(std::make_shared<litepgo::fem::PointwiseVolumetricAdapter<fem::NeoHookean>>(), inputs, outputs,
                 geometry, displacement, plastic, 1);
    Force batched(std::make_shared<litepgo::fem::PointwiseVolumetricAdapter<fem::NeoHookean>>(), inputs, outputs,
                  geometry, displacement, plastic, batch_size);
    const Vec truth = Eigen::Vector2d(2, 3); // lambda, mu
    using Energy = fem::VolumetricFemEnergy<fem::TetP1Basis, fem::TetCentroidQuadrature>;
    const Energy energy(
        std::make_shared<litepgo::fem::PointwiseVolumetricAdapter<fem::NeoHookean>>(), outputs, geometry, plastic,
        std::vector<std::shared_ptr<litepgo::energy::Response>>(
            topology.num_cells(),
            std::make_shared<litepgo::energy::AffineResponse>(litepgo::SpMat((truth).size(), 3), truth)));
    const Vec observed = energy.evaluate(displacement)->gradient();
    const Eigen::MatrixXd J = single.evaluate(truth)->jacobian();
    if (!single.evaluate(truth)->value().isApprox(observed) || !batched.evaluate(truth)->value().isApprox(observed) ||
        !Eigen::MatrixXd(batched.evaluate(truth)->jacobian()).isApprox(J) || J.topRows(15).fullPivLu().rank() != 2)
        throw std::runtime_error("Force, assembly or identifiability check failed");
    solver::GaussNewtonOptions options;
    options.gradient_tolerance = 1e-10;
    Vec recovered;
    for (energy::Response* force :
         {static_cast<energy::Response*>(&single), static_cast<energy::Response*>(&batched)}) {
        ForceResidual residual(*force, observed);
        const auto fit = solver::gauss_newton_solve(residual, Eigen::Vector2d(.7, 1.2), options);
        if (fit.status != solver::GaussNewtonStatus::Converged || (fit.x - truth).norm() > 1e-7)
            throw std::runtime_error("Material recovery failed");
        recovered = fit.x;
    }
    std::cout << std::fixed << std::setprecision(6) << "material fit: lambda=" << recovered[0]
              << ", mu=" << recovered[1] << "\n";
    if (repeats) {
        std::cout << "elements=" << 2 * copies << ", batch_size=" << batch_size << ", repeats=" << repeats << '\n';
        for (bool need : {false, true}) {
            std::vector<double> times;
            for (energy::Response* force :
                 {static_cast<energy::Response*>(&single), static_cast<energy::Response*>(&batched)}) {
                SpMat jacobian = Eigen::MatrixXd::Ones(force->num_outputs(), force->num_inputs()).sparseView();
                for (int i = 0; i < 3; ++i) {
                    auto evaluation = force->evaluate(truth);
                    if (need)
                        jacobian = evaluation->jacobian();
                    evaluation->value();
                }
                std::vector<double> samples;
                for (int i = 0; i < repeats; ++i) {
                    const auto start = std::chrono::steady_clock::now();
                    auto evaluation = force->evaluate(truth);
                    if (need)
                        jacobian = evaluation->jacobian();
                    evaluation->value();
                    samples.push_back(
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
                }
                std::sort(samples.begin(), samples.end());
                times.push_back((samples[(samples.size() - 1) / 2] + samples[samples.size() / 2]) / 2);
            }
            std::cout << (need ? "value+J" : "value") << ": batch1=" << times[0] << " ms, batchN=" << times[1]
                      << " ms, speedup=" << times[0] / times[1] << "x\n";
        }
    }
}
