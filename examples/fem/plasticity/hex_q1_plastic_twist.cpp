#include "litepgo/fem/material/volumetric/pointwise_volumetric_adapter.hpp"
#include "litepgo/fem/response/interpolated_response.hpp"
#include <memory>
// release and clamp a hex bar with a plastic twist Fp.
// Build: cmake --build build/release --target hex_q1_plastic_twist
// Run: build/release/examples/fem/plasticity/hex_q1_plastic_twist

#include "litepgo/energy/response/affine_response.hpp"
#include "litepgo/fem/assembly/volumetric_fem_energy.hpp"
#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/quadrature/3d/gauss_legendre_3d.hpp"
#include "litepgo/fem/response/deformation_gradient_response.hpp"
#include "litepgo/linalg/sparse_solver/auto_sparse_solver.hpp"
#include "litepgo/mesh/extract_surface.hpp"
#include "litepgo/mesh/io/obj.hpp"
#include "litepgo/mesh/io/veg.hpp"
#include "litepgo/solver/newton.hpp"

#include <cmath>
#include <iostream>
#include <numbers>
#include <utility>
#include <vector>

int main(int argc, char** argv) {
    using namespace litepgo;
    const auto mesh = std::get<mesh::HexMesh>(mesh::read_veg(argc > 1 ? argv[1] : LITEPGO_ASSETS_DIR "/box-hex.veg"));
    const auto& topology = mesh.topology;
    const Vec& rest_positions = mesh.positions;
    const Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> rest_nodes(
        rest_positions.data(), topology.num_vertices(), 3);
    if (topology.num_cells() == 0)
        throw std::runtime_error("Example requires a nonempty mesh");

    // Plastic twist: u_p is the displacement a bar twisted past yield keeps once released, here a twist
    // about the vertical center axis growing linearly to 90 degrees at the top. Fp = I + du_p/dX, the
    // gradient of X + u_p, is a rotation times a shear (det Fp = 1), not a rotation. Released, the bar
    // takes the shape X + u_p stress-free; held at both ends, it cannot and keeps residual stress.
    const Eigen::Vector3d lower = rest_nodes.colwise().minCoeff().transpose();
    const Eigen::Vector3d upper = rest_nodes.colwise().maxCoeff().transpose();
    const Eigen::Vector3d center = (lower + upper) / 2;
    const double height = upper.z() - lower.z();
    if (!(height > 0))
        throw std::runtime_error("Example requires nonzero height");
    Vec plastic_displacement(3 * Eigen::Index(topology.num_vertices()));
    for (int v = 0; v < topology.num_vertices(); ++v) {
        const Eigen::Vector3d r = rest_nodes.row(v).transpose() - center;
        const double angle = std::numbers::pi / 2 * (rest_nodes(v, 2) - lower.z()) / height;
        const double c = std::cos(angle), s = std::sin(angle);
        plastic_displacement.segment<3>(3 * Eigen::Index(v)) =
            Eigen::Vector3d(c * r.x() - s * r.y() - r.x(), s * r.x() + c * r.y() - r.y(), 0);
    }

    // Prescribed plastic deformation is a spatial Response; FEM snapshots it at construction.
    const dof::VertexDofMap rest_position_map(topology, 3);

    std::vector<std::shared_ptr<litepgo::energy::Response>> rest_position_fields, plastic_fields;
    rest_position_fields.reserve(topology.num_cells());
    plastic_fields.reserve(topology.num_cells());
    for (int e = 0; e < topology.num_cells(); ++e) {
        rest_position_fields.push_back(
            std::make_shared<fem::InterpolatedResponse<fem::HexQ1Basis>>(rest_position_map.gather(e, rest_positions)));
        plastic_fields.push_back(std::make_shared<fem::InterpolatedResponse<fem::HexQ1Basis>>(
            rest_position_map.gather(e, plastic_displacement)));
    }
    std::vector<std::shared_ptr<litepgo::energy::Response>> Fp_fields;
    Fp_fields.reserve(topology.num_cells());
    for (std::size_t e = 0; e < plastic_fields.size(); ++e)
        Fp_fields.push_back(
            std::make_shared<fem::DeformationGradientResponse>(plastic_fields[e], rest_position_fields[e]));
    auto theta_field = std::make_shared<litepgo::energy::AffineResponse>(litepgo::SpMat(2, 3), Eigen::Vector2d(2, 3));
    fem::VolumetricFemEnergy<fem::HexQ1Basis, fem::GaussLegendre3D<2>> elastic(
        std::make_shared<litepgo::fem::PointwiseVolumetricAdapter<fem::NeoHookean>>(), rest_position_map,
        rest_position_fields, std::move(Fp_fields),
        std::vector<std::shared_ptr<litepgo::energy::Response>>(topology.num_cells(), theta_field));

    const double tolerance = height * 1e-8;
    std::vector<int> bottom, top;
    for (int v = 0; v < topology.num_vertices(); ++v) {
        if (std::abs(rest_nodes(v, 2) - lower.z()) <= tolerance)
            bottom.push_back(v);
        if (std::abs(rest_nodes(v, 2) - upper.z()) <= tolerance)
            top.push_back(v);
    }
    const std::filesystem::path output =
        argc > 2 ? argv[2] : LITEPGO_OUTPUT_DIR "/plastic_twist/hex_q1_plastic_twist.veg";
    if (!output.parent_path().empty())
        std::filesystem::create_directories(output.parent_path());

    // free: clamp the bottom face; the bar twists. clamped: also hold the top face in place;
    // the bar cannot twist and keeps residual stress.
    bool converged = true;
    for (const bool clamp_top : {false, true}) {
        std::vector<int> fixed_dofs;
        for (const auto* face : clamp_top ? std::vector{&bottom, &top} : std::vector{&bottom})
            for (int v : *face)
                for (int c = 0; c < 3; ++c)
                    fixed_dofs.push_back(3 * v + c);
        const dof::DofReduction fixed(elastic.num_dofs(), std::move(fixed_dofs));
        const Vec initial = Vec::Zero(elastic.num_dofs());
        solver::NewtonOptions options;
        options.gradient_tolerance = 1e-8;
        options.max_iterations = 200;
        linalg::AutoSparseSolver solver;
        const auto result = solver::newton_solve(elastic, fixed, initial, options, solver);
        converged = converged && result.status == solver::NewtonStatus::Converged;
        // The top face's twist about the axis, from its vertices' summed in-plane rotation.
        double cross = 0, dot = 0;
        for (int v : top) {
            const Eigen::Vector2d r = (rest_nodes.row(v).transpose() - center).head<2>();
            const Eigen::Vector2d x = r + result.x.segment<2>(3 * Eigen::Index(v));
            cross += r.x() * x.y() - r.y() * x.x();
            dot += r.dot(x);
        }
        std::cout << (clamp_top ? "clamped" : "free")
                  << ": converged=" << (result.status == solver::NewtonStatus::Converged)
                  << ", iterations=" << result.iterations << ", elastic=" << result.energy
                  << ", top twist=" << std::atan2(cross, dot) * 180 / std::numbers::pi << " deg\n";
        const auto path = output.parent_path() / (output.stem().string() + (clamp_top ? "_clamped" : "_free") + ".veg");
        const mesh::HexMesh deformed{rest_positions + result.x, topology};
        mesh::write_veg(path, deformed);
        mesh::write_obj(std::filesystem::path(path).replace_extension(".obj"), mesh::extract_surface(deformed));
    }
    return converged ? 0 : 1;
}
