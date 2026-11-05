#include "litepgo/contact/ipctk_contact_energy.hpp"

#include <ipc/candidates/candidates.hpp>
#include <ipc/ccd/tight_inclusion_ccd.hpp>
#include <ipc/collision_mesh.hpp>
#include <ipc/collisions/normal/normal_collisions.hpp>
#include <ipc/ipc.hpp>
#include <ipc/potentials/barrier_potential.hpp>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace litepgo::contact {

namespace {

IPCTKContactOptions checked_options(const IPCTKContactOptions& options) {
    if (!std::isfinite(options.dhat) || options.dhat <= 0.0) {
        throw std::invalid_argument("IPCTKContactEnergy requires a finite positive dhat");
    }
    if (!std::isfinite(options.min_distance) || options.min_distance < 0.0) {
        throw std::invalid_argument("IPCTKContactEnergy requires a finite non-negative min_distance");
    }
    if (!std::isfinite(options.ccd_tolerance) || options.ccd_tolerance <= 0.0 || options.ccd_max_iterations <= 0) {
        throw std::invalid_argument("IPCTKContactEnergy requires a finite positive CCD tolerance and iteration budget");
    }
    return options;
}

int checked_num_dofs(const mesh::TriangleMesh& surface) {
    surface.validate();
    if (surface.topology.num_vertices() == 0) {
        throw std::invalid_argument("IPCTKContactEnergy requires a nonempty surface");
    }
    return 3 * surface.topology.num_vertices();
}

// The rows of an n x 3 matrix from vertex-major xyz coordinates.
Eigen::MatrixXd vertex_rows(const ConstVecRef& x) {
    return x.reshaped(3, x.size() / 3).transpose();
}

// Each undirected edge of the faces once, as sorted vertex pairs in sorted order.
Eigen::MatrixXi unique_edges(const mesh::TriangleTopology& topology) {
    std::vector<std::array<int, 2>> edges;
    edges.reserve(3 * static_cast<std::size_t>(topology.num_cells()));
    for (int f = 0; f < topology.num_cells(); ++f) {
        const auto& face = topology.cell_vertices(f);
        for (std::size_t j = 0; j < 3; ++j) {
            const int a = face[j];
            const int b = face[(j + 1) % 3];
            edges.push_back({std::min(a, b), std::max(a, b)});
        }
    }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    Eigen::MatrixXi result(static_cast<Eigen::Index>(edges.size()), 2);
    for (std::size_t e = 0; e < edges.size(); ++e) {
        result(static_cast<Eigen::Index>(e), 0) = edges[e][0];
        result(static_cast<Eigen::Index>(e), 1) = edges[e][1];
    }
    return result;
}

Eigen::MatrixXi face_rows(const mesh::TriangleTopology& topology) {
    Eigen::MatrixXi faces(topology.num_cells(), 3);
    for (int f = 0; f < topology.num_cells(); ++f) {
        for (int j = 0; j < 3; ++j) {
            faces(f, j) = topology.cell_vertices(f)[static_cast<std::size_t>(j)];
        }
    }
    return faces;
}

// Adds block (3a.., 3b..) of a pair's local Hessian into the stored entries of
// vertex block (va, vb). The pattern stores whole 3x3 blocks, so each column of
// the block holds three consecutive rows.
void add_block(SpMat& hessian, const ipc::MatrixMax12d& local, int a, int b, int va, int vb) {
    const int* rows = hessian.innerIndexPtr();
    for (int c = 0; c < 3; ++c) {
        const int column = 3 * vb + c;
        const int* begin = rows + hessian.outerIndexPtr()[column];
        const int* end = rows + hessian.outerIndexPtr()[column + 1];
        const int* first = std::lower_bound(begin, end, 3 * va);
        if (end - first < 3 || first[0] != 3 * va || first[2] != 3 * va + 2) {
            throw std::logic_error("IPCTKContactEnergy: an active contact pair is outside the current Hessian "
                                   "pattern; call update_pattern(x) before assembling a Hessian at x");
        }
        double* values = hessian.valuePtr() + (first - rows);
        for (int r = 0; r < 3; ++r) {
            values[r] += local(3 * a + r, 3 * b + c);
        }
    }
}

} // namespace

struct IPCTKContactEnergy::Impl {
    ipc::CollisionMesh mesh;
    ipc::BarrierPotential barrier;
    ipc::TightInclusionCCD ccd;
    double dhat;
    double min_distance;
    bool use_area_weighting;

    ipc::NormalCollisions collisions(const Eigen::MatrixXd& V) const {
        ipc::NormalCollisions result;
        result.set_use_area_weighting(use_area_weighting);
        result.build(mesh, V, dhat, min_distance);
        const auto by_vertices = [this](auto& list) {
            std::stable_sort(list.begin(), list.end(), [this](const auto& p, const auto& q) {
                return p.vertex_ids(mesh.edges(), mesh.faces()) < q.vertex_ids(mesh.edges(), mesh.faces());
            });
        };
        by_vertices(result.vv_collisions);
        by_vertices(result.ev_collisions);
        by_vertices(result.ee_collisions);
        by_vertices(result.fv_collisions);
        return result;
    }

    // The smallest squared distance among the active pairs (+infinity if none).
    double min_squared_distance(const ipc::NormalCollisions& active, const Eigen::MatrixXd& V) const {
        double result = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < active.size(); ++i) {
            result = std::min(result, active[i].compute_distance(V, mesh.edges(), mesh.faces()));
        }
        return result;
    }
};

IPCTKContactEnergy::IPCTKContactEnergy(const mesh::TriangleMesh& surface, IPCTKContactOptions options)
    : DynamicPatternEnergy(checked_num_dofs(surface),
                           SpMat(3 * surface.topology.num_vertices(), 3 * surface.topology.num_vertices())),
      rest_positions_(surface.positions), options_(checked_options(options)) {
    impl_ = std::make_unique<const Impl>(Impl{
        ipc::CollisionMesh(vertex_rows(surface.positions), unique_edges(surface.topology), face_rows(surface.topology)),
        ipc::BarrierPotential(options_.dhat, 1.0),
        ipc::TightInclusionCCD(options_.ccd_tolerance, options_.ccd_max_iterations), options_.dhat,
        options_.min_distance, options_.use_area_weighting});
}

IPCTKContactEnergy::~IPCTKContactEnergy() = default;

int IPCTKContactEnergy::num_active_pairs(const ConstVecRef& x) const {
    if (x.size() != num_dofs() || !x.allFinite()) {
        throw std::invalid_argument("IPCTKContactEnergy::num_active_pairs requires num_dofs finite entries");
    }
    return static_cast<int>(impl_->collisions(vertex_rows(x)).size());
}

double IPCTKContactEnergy::min_active_distance(const ConstVecRef& x) const {
    if (x.size() != num_dofs() || !x.allFinite()) {
        throw std::invalid_argument("IPCTKContactEnergy::min_active_distance requires num_dofs finite entries");
    }
    const Eigen::MatrixXd V = vertex_rows(x);
    return std::sqrt(impl_->min_squared_distance(impl_->collisions(V), V));
}

SpMat IPCTKContactEnergy::pattern_at(const ConstVecRef& x) const {
    if (!x.allFinite()) {
        throw std::invalid_argument("IPCTKContactEnergy requires finite positions");
    }
    const ipc::NormalCollisions active = impl_->collisions(vertex_rows(x));
    std::vector<Triplet> entries;
    for (std::size_t i = 0; i < active.size(); ++i) {
        const auto ids = active[i].vertex_ids(impl_->mesh.edges(), impl_->mesh.faces());
        const int count = active[i].num_vertices();
        for (int a = 0; a < count; ++a) {
            for (int b = 0; b < count; ++b) {
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        entries.emplace_back(3 * ids[static_cast<std::size_t>(a)] + r,
                                             3 * ids[static_cast<std::size_t>(b)] + c, 1.0);
                    }
                }
            }
        }
    }
    SpMat pattern(num_dofs(), num_dofs());
    pattern.setFromTriplets(entries.begin(), entries.end());
    return pattern;
}

double IPCTKContactEnergy::eval_values(const ConstVecRef& x, Vec* gradient, SpMat* hessian,
                                       energy::HessianKind kind) const {
    if (!x.allFinite()) {
        throw std::invalid_argument("IPCTKContactEnergy requires finite positions");
    }
    const Impl& impl = *impl_;
    const Eigen::MatrixXd V = vertex_rows(x);
    const ipc::NormalCollisions active = impl.collisions(V);
    const Eigen::MatrixXi& edges = impl.mesh.edges();
    const Eigen::MatrixXi& faces = impl.mesh.faces();
    const ipc::PSDProjectionMethod projection =
        kind == energy::HessianKind::Projected ? ipc::PSDProjectionMethod::CLAMP : ipc::PSDProjectionMethod::NONE;
    const double min_squared = impl.min_distance * impl.min_distance;

    const std::size_t count = active.size();
    std::vector<double> values(count);
    std::vector<ipc::VectorMax12d> gradients(gradient ? count : 0);
    std::vector<ipc::MatrixMax12d> hessians(hessian ? count : 0);
    tbb::parallel_for(std::size_t{0}, count, [&](std::size_t i) {
        const ipc::NormalCollision& pair = active[i];
        const ipc::VectorMax12d positions = pair.dof(V, edges, faces);
        if (!(pair.compute_distance(positions) > min_squared)) {
            throw std::domain_error("IPCTKContactEnergy: a contact pair is within min_distance");
        }
        values[i] = impl.barrier(pair, positions);
        if (gradient) {
            gradients[i] = impl.barrier.gradient(pair, positions);
        }
        if (hessian) {
            hessians[i] = impl.barrier.hessian(pair, positions, projection);
        }
    });

    // Accumulate in pair order, so the result does not depend on scheduling.
    double value = 0.0;
    if (gradient) {
        gradient->setZero();
    }
    if (hessian) {
        std::fill_n(hessian->valuePtr(), hessian->nonZeros(), 0.0);
    }
    for (std::size_t i = 0; i < count; ++i) {
        const auto ids = active[i].vertex_ids(edges, faces);
        const int n = active[i].num_vertices();
        value += values[i];
        if (gradient) {
            if (!gradients[i].allFinite()) {
                throw std::domain_error("IPCTKContactEnergy: non-finite contact gradient");
            }
            for (int a = 0; a < n; ++a) {
                gradient->segment<3>(3 * ids[static_cast<std::size_t>(a)]) += gradients[i].segment<3>(3 * a);
            }
        }
        if (hessian) {
            if (!hessians[i].allFinite()) {
                throw std::domain_error("IPCTKContactEnergy: non-finite contact Hessian");
            }
            for (int a = 0; a < n; ++a) {
                for (int b = 0; b < n; ++b) {
                    add_block(*hessian, hessians[i], a, b, ids[static_cast<std::size_t>(a)],
                              ids[static_cast<std::size_t>(b)]);
                }
            }
        }
    }
    if (!std::isfinite(value)) {
        throw std::domain_error("IPCTKContactEnergy: non-finite contact energy");
    }
    return value;
}

double IPCTKContactEnergy::feasible_step(const ConstVecRef& x, const ConstVecRef& dx) const {
    if (!x.allFinite() || !dx.allFinite()) {
        throw std::invalid_argument("IPCTKContactEnergy requires finite positions and position deltas");
    }
    const Impl& impl = *impl_;
    const Eigen::MatrixXd V0 = vertex_rows(x);
    if (!(impl.min_squared_distance(impl.collisions(V0), V0) > impl.min_distance * impl.min_distance)) {
        throw std::domain_error("IPCTKContactEnergy: a contact pair is within min_distance");
    }
    if (ipc::has_intersections(impl.mesh, V0)) {
        throw std::domain_error("IPCTKContactEnergy: the surface intersects itself");
    }
    const Eigen::MatrixXd V1 = vertex_rows(x + dx);
    // The candidates' CCD, as in ipc::compute_collision_free_stepsize, except that
    // every query searches all of [0, 1]. IPC Toolkit narrows each query to the
    // earliest impact found so far by other threads, and Tight-Inclusion's
    // conservative answer depends on that bound, so its step varies between runs.
    ipc::Candidates candidates;
    candidates.build(impl.mesh, V0, V1, 0.5 * impl.min_distance);
    const Eigen::MatrixXi& edges = impl.mesh.edges();
    const Eigen::MatrixXi& faces = impl.mesh.faces();
    double step = tbb::parallel_reduce(
        tbb::blocked_range<std::size_t>(0, candidates.size()), 1.0,
        [&](const tbb::blocked_range<std::size_t>& range, double earliest) {
            for (std::size_t i = range.begin(); i < range.end(); ++i) {
                const ipc::CollisionStencil& candidate = candidates[i];
                double toi = 1.0;
                if (candidate.ccd(candidate.dof(V0, edges, faces), candidate.dof(V1, edges, faces), toi,
                                  impl.min_distance, 1.0, impl.ccd)) {
                    earliest = std::min(earliest, toi);
                }
            }
            return earliest;
        },
        [](double a, double b) { return std::min(a, b); });
    return std::clamp(step, 0.0, 1.0);
}

} // namespace litepgo::contact
