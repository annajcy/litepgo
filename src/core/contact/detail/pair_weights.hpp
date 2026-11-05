#pragma once

#include "litepgo/contact/contact_pair_weights.hpp"
#include <cmath>
#include <optional>
#include <stdexcept>
#include <utility>

namespace litepgo::contact::detail {

class PairWeights {
  public:
    PairWeights(std::optional<ContactPairWeights> data, std::size_t nv, std::size_t ne, std::size_t nf)
        : data_(std::move(data)) {
        if (!data_)
            return;
        const auto& w = data_->weights;
        if (w.rows() == 0 || w.rows() != w.cols() || !w.allFinite() || (w.array() < 0).any() ||
            !(w.array() == w.transpose().array()).all())
            throw std::invalid_argument("ContactPairWeights requires a nonempty symmetric finite nonnegative table");
        const auto check = [&](const auto& groups, std::size_t count) {
            if (groups.size() != count)
                throw std::invalid_argument(
                    "ContactPairWeights group lengths must match vertices, contact_edges and faces");
            for (int group : groups)
                if (group < 0 || group >= w.rows())
                    throw std::invalid_argument("ContactPairWeights group IDs must be in [0, table.rows)");
        };
        check(data_->vertex_groups, nv);
        check(data_->edge_groups, ne);
        check(data_->face_groups, nf);
    }
    bool configured() const {
        return data_.has_value();
    }
    double vertex_vertex(int a, int b) const {
        return data_ ? data_->weights(data_->vertex_groups[a], data_->vertex_groups[b]) : 1;
    }
    double edge_vertex(int e, int v) const {
        return data_ ? data_->weights(data_->edge_groups[e], data_->vertex_groups[v]) : 1;
    }
    double vertex_face(int v, int f) const {
        return data_ ? data_->weights(data_->vertex_groups[v], data_->face_groups[f]) : 1;
    }
    double edge_edge(int a, int b) const {
        return data_ ? data_->weights(data_->edge_groups[a], data_->edge_groups[b]) : 1;
    }

  private:
    std::optional<ContactPairWeights> data_;
};

inline double checked_pair_weight(double weight) {
    if (!(weight > 0) || !std::isfinite(weight))
        throw std::domain_error("contact: unrepresentable weighted candidate contribution");
    return weight;
}

} // namespace litepgo::contact::detail
