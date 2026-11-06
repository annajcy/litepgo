#include "litepgo/contact/broad_phase/spatial_hash.hpp"
#include "litepgo/contact/ipc_contact_energy.hpp"
#include "litepgo/contact/plane_penalty_energy.hpp"
#include "litepgo/dof/surface_kinematics.hpp"
#include "litepgo/dof/vertex_displacement_map.hpp"
#include "litepgo/energy/scalar/affine_mapped_energy.hpp"
#include "litepgo/energy/scalar/weighted_sum_energy.hpp"
#include "litepgo/fem/field/identity_field.hpp"
#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/quadrature/3d/tet_centroid_quadrature.hpp"
#include "litepgo/fem/volumetric_fem_energy.hpp"
#include "litepgo/linalg/sparse_solver/auto_sparse_solver.hpp"
#include "litepgo/mesh/extract_surface.hpp"
#include "litepgo/mesh/io/obj.hpp"
#include "litepgo/mesh/io/veg.hpp"
#include "litepgo/solver/newton.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

int main(int argc, char** argv) {
    using namespace litepgo;
    const auto box = std::get<mesh::TetMesh>(mesh::read_veg(argc > 1 ? argv[1] : LITEPGO_ASSETS_DIR "/box-tet.veg"));
    if (box.topology.num_cells() == 0)
        throw std::runtime_error("Example requires a nonempty mesh");
    const int box_vertices = box.topology.num_vertices();
    const Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> box_nodes(box.positions.data(),
                                                                                                box_vertices, 3);
    const double bottom = box_nodes.col(2).minCoeff();
    const double height = box_nodes.col(2).maxCoeff() - bottom;
    if (!(height > 0))
        throw std::runtime_error("Example requires a box with nonzero height");

    // Two copies of the box, one body each, the upper one a small gap above the lower.
    const double gap = 0.04 * height;
    Vec rest_positions(2 * box.positions.size());
    rest_positions << box.positions, box.positions;
    // A small lateral offset avoids an exactly symmetric contact bifurcation.
    for (int v = box_vertices; v < 2 * box_vertices; ++v)
        rest_positions.segment<3>(3 * v) += Eigen::Vector3d(0.013 * height, 0.017 * height, height + gap);
    std::vector<mesh::TetTopology::Cell> cells;
    for (int body = 0; body < 2; ++body)
        for (int e = 0; e < box.topology.num_cells(); ++e) {
            auto cell = box.topology.cell_vertices(e);
            for (int& v : cell)
                v += body * box_vertices;
            cells.push_back(cell);
        }
    const mesh::TetMesh stack{rest_positions, mesh::TetTopology(2 * box_vertices, std::move(cells))};
    const auto& topology = stack.topology;

    const dof::VertexDofMap rest_position_map(topology, 3);
    using XField = fem::InterpolatedField3D<fem::TetP1Basis>;
    std::vector<XField> rest_position_fields;
    rest_position_fields.reserve(topology.num_cells());
    for (int e = 0; e < topology.num_cells(); ++e)
        rest_position_fields.emplace_back(rest_position_map.gather(e, rest_positions));
    const fem::IdentityField3D Fp_field(Eigen::Matrix3d::Identity().reshaped());
    const fem::IdentityField3D theta_field(Eigen::Vector2d(0.5, 0.75));
    auto elastic =
        std::make_shared<fem::VolumetricFemEnergy<fem::NeoHookean, fem::TetP1Basis, fem::TetCentroidQuadrature, XField,
                                                  fem::IdentityField3D, fem::IdentityField3D>>(
            rest_position_map, std::move(rest_position_fields),
            std::vector<fem::IdentityField3D>(topology.num_cells(), Fp_field),
            std::vector<fem::IdentityField3D>(topology.num_cells(), theta_field));
    const int num_dofs = elastic->num_dofs();

    // Pin the lateral DOFs on the outer faces to remove rigid modes.
    const double tolerance = height * 1e-8;
    std::vector<int> clamped;
    for (int v = 0; v < box_vertices; ++v) {
        if (std::abs(box_nodes(v, 2) - bottom) <= tolerance)
            clamped.insert(clamped.end(), {3 * v, 3 * v + 1});
        if (std::abs(box_nodes(v, 2) - bottom - height) <= tolerance) {
            const int upper = v + box_vertices;
            clamped.insert(clamped.end(), {3 * upper, 3 * upper + 1});
        }
    }

    // Contact acts on the boundary of both boxes, whose vertices move with their
    // volume displacement DOFs. The barrier's weight kappa is its stiffness.
    const auto extracted = mesh::extract_surface_with_vertices(stack);
    const auto& surface = extracted.mesh;
    const SpMat W = dof::build_vertex_displacement_map(extracted.vertices, num_dofs);
    contact::IPCContactOptions contact_options;
    contact_options.dhat = 0.004 * height;
    auto barrier =
        std::make_shared<contact::IPCContactEnergy>(surface, contact_options, std::make_shared<contact::SpatialHash>());
    const dof::SurfaceKinematics kinematics(surface.positions, W);
    const int n = surface.topology.num_vertices();
    auto bottom_plate = std::make_shared<contact::PlanePenaltyEnergy>(n, 2, bottom + 0.2 * height, 20.0);
    auto top_plate = std::make_shared<contact::PlanePenaltyEnergy>(n, 2, bottom + 1.8 * height + gap, 20.0, false);
    const double kappa = 1e7;
    energy::WeightedSumEnergy potential({
        {elastic, 1},
        {std::make_shared<energy::AffineMappedEnergy>(barrier, W, kinematics.rest_positions()), kappa},
        {std::make_shared<energy::AffineMappedEnergy>(bottom_plate, W, kinematics.rest_positions()), 1},
        {std::make_shared<energy::AffineMappedEnergy>(top_plate, W, kinematics.rest_positions()), 1},
    });
    const dof::DofReduction fixed(num_dofs, clamped);
    const Vec initial = Vec::Zero(num_dofs);
    solver::NewtonOptions options;
    options.gradient_tolerance = 1e-8;
    options.max_iterations = 200;
    linalg::AutoSparseSolver solver;
    const auto result = solver::newton_solve(potential, fixed, initial, options, solver);
    const bool converged = result.status == solver::NewtonStatus::Converged;

    const Vec surface_positions = kinematics.positions(result.x);
    const int pairs = barrier->num_active_pairs(surface_positions);
    const double distance = barrier->min_active_distance(surface_positions);
    const double lower_energy = bottom_plate->eval_value(surface_positions);
    const double upper_energy = top_plate->eval_value(surface_positions);
    std::cout << "converged=" << converged << ", iterations=" << result.iterations << ", potential=" << result.energy
              << ", bottom penalty=" << lower_energy << ", top penalty=" << upper_energy << ", active pairs=" << pairs
              << ", min distance=" << distance << '\n';
    const std::filesystem::path output = argc > 2 ? argv[2] : LITEPGO_OUTPUT_DIR "/tet_p1_plate_compression.veg";
    if (!output.parent_path().empty())
        std::filesystem::create_directories(output.parent_path());
    const mesh::TetMesh deformed{rest_positions + result.x, topology};
    mesh::write_veg(output, deformed);
    mesh::write_obj(std::filesystem::path(output).replace_extension(".obj"), mesh::extract_surface(deformed));
    return converged && lower_energy > 0 && upper_energy > 0 && pairs > 0 && distance > 0 &&
                   distance < contact_options.dhat
               ? 0
               : 1;
}
