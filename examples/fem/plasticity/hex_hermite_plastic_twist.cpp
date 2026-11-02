// release and clamp a Hermite hex bar with a plastic twist Fp.
// Build: cmake --build build/release --target hex_hermite_plastic_twist
// Run: build/release/examples/fem/plasticity/hex_hermite_plastic_twist

#include "litepgo/fem/field/deformation_gradient_field.hpp"
#include "litepgo/fem/field/identity_field.hpp"
#include "litepgo/fem/kinematics/basis_dof_transform.hpp"
#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/point_displacement_map.hpp"
#include "litepgo/fem/quadrature/3d/gauss_legendre_3d.hpp"
#include "litepgo/fem/sample_boundary.hpp"
#include "litepgo/fem/volumetric_fem_energy.hpp"
#include "litepgo/linalg/sparse_solver/auto_sparse_solver.hpp"
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
    // u_p uses the Hermite basis, so it is C1 like the displacement. Its 24 physical DOFs per vertex are
    // the twist's value and derivatives u, u_X, u_Y, u_Z, u_XY, u_XZ, u_YZ, u_XYZ; for each Z it is
    // linear in X and Y, so u_XY and u_XYZ vanish.
    const Eigen::Vector3d lower = rest_nodes.colwise().minCoeff().transpose();
    const Eigen::Vector3d upper = rest_nodes.colwise().maxCoeff().transpose();
    const Eigen::Vector3d center = (lower + upper) / 2;
    const double height = upper.z() - lower.z();
    if (!(height > 0))
        throw std::runtime_error("Example requires nonzero height");
    const double rate = std::numbers::pi / 2 / height; // twist angle per unit height
    Vec plastic_dofs(24 * Eigen::Index(topology.num_vertices()));
    for (int v = 0; v < topology.num_vertices(); ++v) {
        const Eigen::Vector2d r = (rest_nodes.row(v).transpose() - center).head<2>();
        const double angle = rate * (rest_nodes(v, 2) - lower.z());
        const double c = std::cos(angle), s = std::sin(angle);
        const Eigen::Vector2d p(c * r.x() - s * r.y(), s * r.x() + c * r.y()); // r rotated by angle
        Eigen::Matrix<double, 3, 8> slots = Eigen::Matrix<double, 3, 8>::Zero();
        slots.col(0).head<2>() = p - r;
        slots.col(1).head<2>() = Eigen::Vector2d(c - 1, s);
        slots.col(2).head<2>() = Eigen::Vector2d(-s, c - 1);
        slots.col(3).head<2>() = rate * Eigen::Vector2d(-p.y(), p.x());
        slots.col(5).head<2>() = rate * Eigen::Vector2d(-s, c);
        slots.col(6).head<2>() = rate * Eigen::Vector2d(-c, -s);
        plastic_dofs.segment<24>(24 * Eigen::Index(v)) = slots.reshaped();
    }

    // DeformationGradientField3D(u_p, X) borrows both fields, so both vectors are filled before it takes
    // references and outlive the energy, which keeps its own copy of the rest fields. BasisDofTransform
    // converts each element's physical Hermite DOFs to basis coefficients, as the energy does for u.
    const dof::VertexDofMap rest_position_map(topology, 3), displacement_map(topology, 24);
    using XField = fem::InterpolatedField3D<fem::HexQ1Basis>;
    using PlasticField = fem::InterpolatedField3D<fem::HexHermiteBasis>;
    std::vector<XField> rest_position_fields;
    std::vector<PlasticField> plastic_fields;
    rest_position_fields.reserve(topology.num_cells());
    plastic_fields.reserve(topology.num_cells());
    for (int e = 0; e < topology.num_cells(); ++e) {
        rest_position_fields.emplace_back(rest_position_map.gather(e, rest_positions));
        const fem::BasisDofTransform<fem::HexHermiteBasis> transform(rest_position_fields.back());
        plastic_fields.emplace_back(transform.to_coefficients(displacement_map.gather(e, plastic_dofs)));
    }
    std::vector<fem::DeformationGradientField3D> Fp_fields;
    Fp_fields.reserve(topology.num_cells());
    for (std::size_t e = 0; e < plastic_fields.size(); ++e)
        Fp_fields.emplace_back(plastic_fields[e], rest_position_fields[e]);
    const fem::IdentityField3D theta_field(Eigen::Vector2d(2, 3));
    fem::VolumetricFemEnergy<fem::NeoHookean, fem::HexHermiteBasis, fem::GaussLegendre3D<4>, XField,
                             fem::DeformationGradientField3D, fem::IdentityField3D>
        elastic(displacement_map, rest_position_fields, std::move(Fp_fields),
                std::vector<fem::IdentityField3D>(topology.num_cells(), theta_field));

    const double tolerance = height * 1e-8;
    std::vector<int> bottom, top;
    for (int v = 0; v < topology.num_vertices(); ++v) {
        if (std::abs(rest_nodes(v, 2) - lower.z()) <= tolerance)
            bottom.push_back(v);
        if (std::abs(rest_nodes(v, 2) - upper.z()) <= tolerance)
            top.push_back(v);
    }
    const std::filesystem::path output =
        argc > 2 ? argv[2] : LITEPGO_OUTPUT_DIR "/plastic_twist/hex_hermite_plastic_twist.veg";
    if (!output.parent_path().empty())
        std::filesystem::create_directories(output.parent_path());

    const auto surface = fem::sample_boundary(topology, rest_position_fields, 4);
    const auto W = fem::build_point_displacement_map<fem::HexHermiteBasis>(surface.embedding, displacement_map,
                                                                           rest_position_fields);

    // free: clamp the bottom face; the bar twists. clamped: also hold the top face in place;
    // the bar cannot twist and keeps residual stress. Clamping a face z = const fixes u and its
    // in-face derivatives u_X, u_Y, u_XY, which determine the displacement on the whole face.
    bool converged = true;
    for (const bool clamp_top : {false, true}) {
        std::vector<int> fixed_dofs;
        for (const auto* face : clamp_top ? std::vector{&bottom, &top} : std::vector{&bottom})
            for (int v : *face)
                for (int slot : {0, 1, 2, 4})
                    for (int c = 0; c < 3; ++c)
                        fixed_dofs.push_back(24 * v + 3 * slot + c);
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
            const Eigen::Vector2d x = r + result.x.segment<2>(24 * Eigen::Index(v));
            cross += r.x() * x.y() - r.y() * x.x();
            dot += r.dot(x);
        }
        std::cout << (clamp_top ? "clamped" : "free")
                  << ": converged=" << (result.status == solver::NewtonStatus::Converged)
                  << ", iterations=" << result.iterations << ", elastic=" << result.energy
                  << ", free residual=" << result.gradient_norm
                  << ", top twist=" << std::atan2(cross, dot) * 180 / std::numbers::pi << " deg\n";
        const auto path = output.parent_path() / (output.stem().string() + (clamp_top ? "_clamped" : "_free") + ".veg");
        Vec deformed_positions = rest_positions;
        for (int v = 0; v < topology.num_vertices(); ++v)
            deformed_positions.segment<3>(3 * Eigen::Index(v)) += result.x.segment<3>(24 * Eigen::Index(v));
        const mesh::HexMesh deformed{std::move(deformed_positions), topology};
        mesh::write_veg(path, deformed);
        if (!converged)
            throw std::runtime_error("Hermite equilibrium did not converge");
        mesh::write_obj(std::filesystem::path(path).replace_extension(".obj"),
                        mesh::TriangleMesh{surface.mesh.positions + W * result.x, surface.mesh.topology});
    }
    return converged ? 0 : 1;
}
