#include "litepgo/energy/vertex_pulling_energy.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace litepgo::energy {

namespace {

// The diagonal pattern on the pulled DOFs.
SpMat diagonal_pattern(int num_dofs, const std::vector<int>& dofs) {
    std::vector<Triplet> entries;
    entries.reserve(dofs.size());
    for (const int dof : dofs) {
        entries.emplace_back(dof, dof, 1.0);
    }
    SpMat pattern(num_dofs, num_dofs);
    pattern.setFromTriplets(entries.begin(), entries.end());
    return pattern;
}

} // namespace

VertexPullingEnergy::Setup VertexPullingEnergy::checked_setup(int num_dofs,
                                                              const std::vector<std::array<int, 3>>& position_dofs,
                                                              Vec targets, double stiffness) {
    if (num_dofs < 0) {
        throw std::invalid_argument("VertexPullingEnergy requires a non-negative DOF count");
    }
    if (!std::isfinite(stiffness) || stiffness <= 0.0) {
        throw std::invalid_argument("VertexPullingEnergy requires a finite positive stiffness");
    }
    if (targets.size() != 3 * static_cast<Eigen::Index>(position_dofs.size()) || !targets.allFinite()) {
        throw std::invalid_argument("VertexPullingEnergy requires three finite target entries per pulled vertex");
    }
    std::vector<int> dofs;
    dofs.reserve(3 * position_dofs.size());
    for (const auto& vertex : position_dofs) {
        dofs.insert(dofs.end(), vertex.begin(), vertex.end());
    }
    std::vector<int> sorted = dofs;
    std::sort(sorted.begin(), sorted.end());
    if (!sorted.empty() && (sorted.front() < 0 || sorted.back() >= num_dofs)) {
        throw std::invalid_argument("VertexPullingEnergy position DOFs must be in [0, num_dofs)");
    }
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
        throw std::invalid_argument("VertexPullingEnergy position DOFs must be distinct");
    }
    return {num_dofs, std::move(dofs), std::move(targets), stiffness};
}

VertexPullingEnergy::VertexPullingEnergy(int num_dofs, std::vector<std::array<int, 3>> position_dofs, Vec targets,
                                         double stiffness)
    : VertexPullingEnergy(checked_setup(num_dofs, position_dofs, std::move(targets), stiffness)) {}

VertexPullingEnergy::VertexPullingEnergy(Setup setup)
    : FixedPatternEnergy(setup.num_dofs, diagonal_pattern(setup.num_dofs, setup.dofs)), dofs_(std::move(setup.dofs)),
      targets_(std::move(setup.targets)), stiffness_(setup.stiffness) {}

double VertexPullingEnergy::eval_values(const ConstVecRef& x, Vec* gradient, SpMat* hessian, HessianKind) const {
    if (gradient != nullptr) {
        gradient->setZero();
    }
    double squared_distance = 0.0;
    for (std::size_t i = 0; i < dofs_.size(); ++i) {
        const double gap = x[dofs_[i]] - targets_[static_cast<Eigen::Index>(i)];
        squared_distance += gap * gap;
        if (gradient != nullptr) {
            (*gradient)[dofs_[i]] = stiffness_ * gap;
        }
    }
    if (hessian != nullptr) {
        // One stored entry per pulled DOF, all equal to the stiffness.
        std::fill_n(hessian->valuePtr(), hessian->nonZeros(), stiffness_);
    }
    return 0.5 * stiffness_ * squared_distance;
}

} // namespace litepgo::energy
