#include "litepgo/response/mesh_response.hpp"
#include <algorithm>
#include <limits>
#include <map>
#include <tbb/parallel_for.h>

namespace litepgo::response {
MeshResponse::MeshResponse(dof::ResponseAssembler assembler, Callback callback, int batch_size)
    : FixedPatternResponse(assembler.jacobian_pattern()), assembler_(std::move(assembler)),
      callback_(std::move(callback)) {
    if (batch_size <= 0 || !callback_)
        throw std::invalid_argument("MeshResponse: invalid batch size or callback");
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
        data.values = callback_(ids, inputs, J ? &jacobians : nullptr);
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
} // namespace litepgo::response
