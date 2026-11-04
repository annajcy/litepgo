#include "litepgo/energy/response/mesh_response.hpp"
#include <algorithm>
#include <limits>
#include <map>
#include <tbb/parallel_for.h>

namespace litepgo::energy {
MeshResponse::MeshResponse(dof::ResponseAssembler assembler, int batch_size)
    : FixedPatternResponse(assembler.jacobian_pattern()), assembler_(std::move(assembler)) {
    if (batch_size <= 0)
        throw std::invalid_argument("MeshResponse: invalid batch size");
    std::map<std::pair<int, int>, std::vector<int>> groups;
    for (int e = 0; e < assembler_.num_elements(); ++e)
        groups[{static_cast<int>(assembler_.input_indices(e).size()),
                static_cast<int>(assembler_.output_indices(e).size())}]
            .push_back(e);
    for (const auto& [shape, ids] : groups)
        for (std::size_t begin = 0; begin < ids.size(); begin += batch_size) {
            const auto end = std::min(ids.size(), begin + batch_size);
            batches_.emplace_back(ids.begin() + begin, ids.begin() + end);
        }
    storage_.resize(batches_.size());
    views_.resize(assembler_.num_elements());
}

Vec MeshResponse::eval_values(const ConstVecRef& x, SpMat* J) const {
    tbb::parallel_for(std::size_t(0), batches_.size(), [&](std::size_t k) {
        const auto& ids = batches_[k];
        const Eigen::Index count = ids.size();
        const Eigen::Index input_size = assembler_.input_indices(ids.front()).size();
        const Eigen::Index output_size = assembler_.output_indices(ids.front()).size();
        auto& data = storage_[k];
        auto& inputs = data.inputs;
        inputs.resize(count, input_size);
        for (Eigen::Index b = 0; b < count; ++b) {
            const auto indices = assembler_.input_indices(ids[b]);
            for (Eigen::Index c = 0; c < input_size; ++c)
                inputs(b, c) = x[indices[c]];
        }
        auto& jacobians = data.jacobian;
        if (J)
            jacobians.setConstant(count * output_size, input_size, std::numeric_limits<double>::quiet_NaN());
        data.values = eval_batch(ids, inputs, J ? &jacobians : nullptr);
        const auto& values = data.values;
        if (values.rows() != count || values.cols() != output_size || !values.allFinite() ||
            (J &&
             (jacobians.rows() != count * output_size || jacobians.cols() != input_size || !jacobians.allFinite())))
            throw std::logic_error("MeshResponse: invalid batch outputs");
        for (Eigen::Index b = 0; b < count; ++b) {
            // Empty blocks use null pointers to avoid arithmetic on empty storage.
            const Eigen::Map<const Vec> value(output_size ? values.data() + b * output_size : nullptr, output_size);
            const Eigen::Map<const BatchMatrix> jacobian(
                J && output_size && input_size ? data.jacobian.data() + b * output_size * input_size : nullptr,
                J ? output_size : 0, J ? input_size : 0);
            views_[ids[b]] = dof::ResponseAssembler::ElementView(value, jacobian);
        }
    });
    Vec result;
    assembler_.assemble(views_, result, J);
    return result;
}
Vec MeshResponse::eval_jvp_values(const ConstVecRef& x, const ConstVecRef& dx) const {
    return product(x, dx, false);
}
Vec MeshResponse::eval_vjp_values(const ConstVecRef& x, const ConstVecRef& gy) const {
    return product(x, gy, true);
}
Vec MeshResponse::product(const ConstVecRef& x, const ConstVecRef& direction, bool transpose) const {
    std::vector<BatchMatrix> products(batches_.size());
    tbb::parallel_for(std::size_t(0), batches_.size(), [&](std::size_t k) {
        const auto& ids = batches_[k];
        const auto ni = assembler_.input_indices(ids.front()).size();
        const auto no = assembler_.output_indices(ids.front()).size();
        BatchMatrix local_x(ids.size(), ni), local_d(ids.size(), transpose ? no : ni);
        for (std::size_t b = 0; b < ids.size(); ++b) {
            const auto in = assembler_.input_indices(ids[b]);
            const auto directions = transpose ? assembler_.output_indices(ids[b]) : in;
            for (std::size_t c = 0; c < in.size(); ++c)
                local_x(b, c) = x[in[c]];
            for (std::size_t c = 0; c < directions.size(); ++c)
                local_d(b, c) = direction[directions[c]];
        }
        products[k] = transpose ? eval_vjp_batch(ids, local_x, local_d) : eval_jvp_batch(ids, local_x, local_d);
        if (products[k].rows() != static_cast<Eigen::Index>(ids.size()) ||
            products[k].cols() != static_cast<Eigen::Index>(transpose ? ni : no) || !products[k].allFinite())
            throw std::logic_error("MeshResponse: invalid batch product");
    });
    Vec result = Vec::Zero(transpose ? num_inputs() : num_outputs());
    // Reduce in original element order, including repeated indices.
    std::vector<std::pair<std::size_t, std::size_t>> locations(assembler_.num_elements());
    for (std::size_t k = 0; k < batches_.size(); ++k)
        for (std::size_t b = 0; b < batches_[k].size(); ++b)
            locations[batches_[k][b]] = {k, b};
    for (int e = 0; e < assembler_.num_elements(); ++e) {
        const auto [k, b] = locations[e];
        const auto indices = transpose ? assembler_.input_indices(e) : assembler_.output_indices(e);
        for (std::size_t c = 0; c < indices.size(); ++c)
            result[indices[c]] += products[k](b, c);
    }
    return result;
}
void MeshResponse::check_batch_product(std::span<const int> ids, const BatchMatrix& x, const BatchMatrix& d,
                                       bool transpose) const {
    if (ids.empty() || x.rows() != static_cast<Eigen::Index>(ids.size()) || d.rows() != x.rows() || !x.allFinite() ||
        !d.allFinite())
        throw std::invalid_argument("MeshResponse: invalid batch product inputs");
    for (int id : ids) {
        if (id < 0 || id >= assembler_.num_elements())
            throw std::invalid_argument("MeshResponse: invalid element ID");
        const auto ni = assembler_.input_indices(id).size();
        const auto no = assembler_.output_indices(id).size();
        if (x.cols() != static_cast<Eigen::Index>(ni) || d.cols() != static_cast<Eigen::Index>(transpose ? no : ni) ||
            no != assembler_.output_indices(ids.front()).size())
            throw std::invalid_argument("MeshResponse: mismatched batch dimensions");
    }
}
BatchMatrix MeshResponse::eval_jvp_batch(std::span<const int> ids, const BatchMatrix& x, const BatchMatrix& d) const {
    check_batch_product(ids, x, d, false);
    const Eigen::Index no = assembler_.output_indices(ids.front()).size();
    BatchMatrix J = BatchMatrix::Constant(x.rows() * no, x.cols(), std::numeric_limits<double>::quiet_NaN());
    const BatchMatrix value = eval_batch(ids, x, &J);
    if (value.rows() != x.rows() || value.cols() != no || !value.allFinite() || J.rows() != x.rows() * no ||
        J.cols() != x.cols() || !J.allFinite())
        throw std::logic_error("MeshResponse: invalid batch Jacobian");
    BatchMatrix result(x.rows(), no);
    for (Eigen::Index b = 0; b < x.rows(); ++b)
        result.row(b) = (J.middleRows(b * no, no) * d.row(b).transpose()).transpose();
    return result;
}
BatchMatrix MeshResponse::eval_vjp_batch(std::span<const int> ids, const BatchMatrix& x, const BatchMatrix& d) const {
    check_batch_product(ids, x, d, true);
    const Eigen::Index no = assembler_.output_indices(ids.front()).size();
    BatchMatrix J = BatchMatrix::Constant(x.rows() * no, x.cols(), std::numeric_limits<double>::quiet_NaN());
    const BatchMatrix value = eval_batch(ids, x, &J);
    if (value.rows() != x.rows() || value.cols() != no || !value.allFinite() || J.rows() != x.rows() * no ||
        J.cols() != x.cols() || !J.allFinite())
        throw std::logic_error("MeshResponse: invalid batch Jacobian");
    BatchMatrix result(x.rows(), x.cols());
    for (Eigen::Index b = 0; b < x.rows(); ++b)
        result.row(b) = (J.middleRows(b * no, no).transpose() * d.row(b).transpose()).transpose();
    return result;
}
} // namespace litepgo::energy
