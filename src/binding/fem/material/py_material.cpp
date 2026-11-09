#include "fem/material/py_material.hpp"
#include "litepgo/fem/kinematics/positive_det_feasible_step.hpp"
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include "detail/templates.hpp"
#include "litepgo/fem/material/shell/koiter_stvk.hpp"
#include "litepgo/fem/material/shell/pointwise_shell_adapter.hpp"
#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/material/volumetric/pointwise_volumetric_adapter.hpp"
#include <nanobind/stl/vector.h>
#include <string>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::fem::binding {
namespace {

using Input = nb::ndarray<const double, nb::c_contig, nb::device::cpu>;

// A new C-order float64 NumPy array owning its storage; fill(data) writes every entry.
template <class Fill> nb::object new_array(std::initializer_list<size_t> shape, Fill&& fill) {
    size_t size = 1;
    for (size_t extent : shape)
        size *= extent;
    auto* data = new double[size == 0 ? 1 : size];
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    fill(data);
    return nb::ndarray<nb::numpy, double>(data, shape, owner).cast();
}

std::string shape_text(std::initializer_list<size_t> shape) {
    std::string text = "(";
    for (auto it = shape.begin(); it != shape.end(); ++it)
        text += (it == shape.begin() ? "" : ", ") + std::to_string(*it);
    return text + (shape.size() == 1 ? ",)" : ")");
}

// The named output as a C-contiguous float64 array of exactly this shape.
Input read_output(nb::handle instance, nb::handle result, const char* name, std::initializer_list<size_t> shape) {
    const std::string owner = nb::cast<std::string>(nb::getattr(instance.type(), "__name__")) +
                              (std::string(name) == "steps" ? ".eval_feasible_step" : ".eval");
    nb::dict outputs = nb::borrow<nb::dict>(result);
    if (!outputs.contains(name))
        throw nb::key_error((owner + " did not return requested output '" + name + "'").c_str());
    nb::object array = nb::module_::import_("numpy").attr("ascontiguousarray")(outputs[name], "dtype"_a = "float64");
    Input values = nb::cast<Input>(array);
    bool valid = values.ndim() == shape.size();
    size_t axis = 0;
    for (size_t extent : shape)
        valid = valid && values.shape(axis++) == extent;
    if (!valid)
        throw nb::value_error((owner + " returned " + name + " with shape " +
                               nb::cast<std::string>(nb::repr(array.attr("shape"))) + "; expected " + shape_text(shape))
                                  .c_str());
    return values;
}

nb::object call_eval(nb::handle instance, nb::tuple arguments, const std::vector<const char*>& names) {
    nb::list requested;
    for (const char* name : names)
        requested.append(nb::str(name));
    nb::object result = instance.attr("eval")(*arguments, "outputs"_a = nb::tuple(requested));
    if (!nb::isinstance<nb::dict>(result))
        throw nb::type_error(
            (nb::cast<std::string>(nb::getattr(instance.type(), "__name__")) + ".eval must return a dict").c_str());
    return result;
}

int read_num_parameters(nb::handle instance) {
    nb::object value = nb::getattr(instance.type(), "num_parameters", nb::none());
    if (!nb::isinstance<nb::int_>(value) || nb::isinstance<nb::bool_>(value) || nb::cast<long long>(value) < 0 ||
        nb::cast<long long>(value) > 1 << 20)
        throw nb::type_error("material class must define a non-negative integer num_parameters");
    if (!PyCallable_Check(nb::getattr(instance, "eval", nb::none()).ptr()))
        throw nb::type_error("material class must define eval(...)");
    return nb::cast<int>(value);
}

// (N,3,3) with [n,i,j] = matrix(i,j) of column n.
nb::object matrices(const Eigen::Matrix<double, 9, Eigen::Dynamic>& packed) {
    const auto count = static_cast<size_t>(packed.cols());
    return new_array({count, 3, 3}, [&](double* data) {
        for (size_t n = 0; n < count; ++n)
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    data[9 * n + 3 * i + j] = packed(i + 3 * j, static_cast<Eigen::Index>(n));
    });
}

// (N, rows) with [n, r] = columns(r, n).
nb::object rows(const Eigen::Ref<const Eigen::MatrixXd>& columns) {
    const auto count = static_cast<size_t>(columns.cols()), width = static_cast<size_t>(columns.rows());
    return new_array({count, width}, [&](double* data) {
        for (size_t n = 0; n < count; ++n)
            for (size_t r = 0; r < width; ++r)
                data[width * n + r] = columns(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(n));
    });
}

nb::object vector(const Vec& values) {
    const auto count = static_cast<size_t>(values.size());
    return new_array({count}, [&](double* data) {
        for (size_t n = 0; n < count; ++n)
            data[n] = values[static_cast<Eigen::Index>(n)];
    });
}

void read_vector(nb::handle instance, nb::handle result, const char* name, size_t count, Vec& out) {
    const Input values = read_output(instance, result, name, {count});
    out = Eigen::Map<const Vec>(values.data(), static_cast<Eigen::Index>(count));
}

// [n, a..., b...] in C order to column n holding a column-major (rows x cols) matrix; the
// row index is (i,j) with i+3*j for 3x3 blocks, or r for packed vectors.
template <class Out>
void read_matrices(nb::handle instance, nb::handle result, const char* name, std::initializer_list<size_t> shape,
                   int row_blocks, int cols, Out& out) {
    const Input values = read_output(instance, result, name, shape);
    const auto count = static_cast<Eigen::Index>(*shape.begin());
    const int rows = row_blocks == 3 ? 9 : row_blocks;
    out.resize(rows * cols, count);
    const double* data = values.data();
    for (Eigen::Index n = 0; n < count; ++n)
        for (int r = 0; r < rows; ++r) {
            // Python row index in C order: (i,j) -> 3*i+j for 3x3 blocks.
            const int source_row = row_blocks == 3 ? 3 * (r % 3) + r / 3 : r;
            for (int c = 0; c < cols; ++c)
                out(r + rows * c, n) = data[(n * rows + source_row) * cols + c];
        }
}

// Columns of a 9 x 9 block also follow vec order; Python's trailing (k,l) axes use 3*k+l.
template <class Out> void permute_matrix_columns(Out& out) {
    for (Eigen::Index n = 0; n < out.cols(); ++n) {
        const Eigen::Matrix<double, 9, 9> block = out.col(n).reshaped(9, 9);
        Eigen::Matrix<double, 9, 9> permuted;
        for (int c = 0; c < 9; ++c)
            permuted.col(c % 3 * 3 + c / 3) = block.col(c);
        out.col(n) = permuted.reshaped();
    }
}
Vec default_feasible_step(const VolumetricMaterialBatch& in, const Eigen::Matrix<double, 9, Eigen::Dynamic>& dF) {
    detail::check_volumetric_batch_inputs(in, static_cast<int>(in.theta.rows()));
    if (dF.cols() != in.size())
        throw std::invalid_argument("Volumetric batch material: incompatible increment dimensions");
    Vec steps(in.size());
    for (Eigen::Index n = 0; n < in.size(); ++n)
        steps[n] = eval_positive_det_feasible_step(in.F.col(n).reshaped(3, 3), dF.col(n).reshaped(3, 3));
    return steps;
}
} // namespace

PyVolumetricMaterial::PyVolumetricMaterial(nb::handle instance)
    : instance_(instance), num_parameters_(read_num_parameters(instance)),
      has_feasible_step_(
          !instance.type().attr("eval_feasible_step").is(nb::type<PyVolumetricMaterial>().attr("eval_feasible_step"))) {
    if (!PyCallable_Check(instance.attr("eval_feasible_step").ptr()))
        throw nb::type_error("material eval_feasible_step must be callable");
}

void PyVolumetricMaterial::eval(const VolumetricMaterialBatch& in, VolumetricMaterialBatchOutputs& out) const {
    nb::gil_scoped_acquire gil;
    const nb::handle instance = instance_;
    const auto count = static_cast<size_t>(in.size()), k = static_cast<size_t>(num_parameters_);
    std::vector<const char*> names;
    if (out.W)
        names.push_back("W");
    if (out.dW_dF)
        names.push_back("dW_dF");
    if (out.d2W_dF2)
        names.push_back("d2W_dF2");
    if (out.d2W_dF_dtheta)
        names.push_back("d2W_dF_dtheta");
    if (out.d2W_dF_dFp)
        names.push_back("d2W_dF_dFp");
    const nb::object result =
        call_eval(instance, nb::make_tuple(matrices(in.F), matrices(in.Fp), rows(in.theta)), names);
    if (out.W)
        read_vector(instance, result, "W", count, *out.W);
    if (out.dW_dF)
        read_matrices(instance, result, "dW_dF", {count, 3, 3}, 3, 1, *out.dW_dF);
    if (out.d2W_dF2) {
        read_matrices(instance, result, "d2W_dF2", {count, 3, 3, 3, 3}, 3, 9, *out.d2W_dF2);
        permute_matrix_columns(*out.d2W_dF2);
    }
    if (out.d2W_dF_dtheta)
        read_matrices(instance, result, "d2W_dF_dtheta", {count, 3, 3, k}, 3, num_parameters_, *out.d2W_dF_dtheta);
    if (out.d2W_dF_dFp) {
        read_matrices(instance, result, "d2W_dF_dFp", {count, 3, 3, 3, 3}, 3, 9, *out.d2W_dF_dFp);
        permute_matrix_columns(*out.d2W_dF_dFp);
    }
}

Vec PyVolumetricMaterial::eval_feasible_step(const VolumetricMaterialBatch& in,
                                             const Eigen::Matrix<double, 9, Eigen::Dynamic>& dF) const {
    if (!has_feasible_step_)
        return default_feasible_step(in, dF);
    const Eigen::Index count = in.size();
    Vec steps(count);
    nb::gil_scoped_acquire gil;
    const nb::handle instance = instance_;
    nb::object result =
        instance.attr("eval_feasible_step")(matrices(in.F), matrices(in.Fp), rows(in.theta), matrices(dF));
    nb::dict wrapped;
    wrapped["steps"] = result;
    read_vector(instance, wrapped, "steps", static_cast<size_t>(count), steps);
    return steps;
}

PyShellMaterial::PyShellMaterial(nb::handle instance)
    : instance_(instance), num_parameters_(read_num_parameters(instance)) {}

void PyShellMaterial::eval(const ShellMaterialBatch& in, ShellMaterialBatchOutputs& out) const {
    nb::gil_scoped_acquire gil;
    const nb::handle instance = instance_;
    const auto count = static_cast<size_t>(in.size()), k = static_cast<size_t>(num_parameters_);
    std::vector<const char*> names;
    if (out.W)
        names.push_back("W");
    if (out.dW_dab)
        names.push_back("dW_dab");
    if (out.d2W_dab2)
        names.push_back("d2W_dab2");
    if (out.d2W_dab_dtheta)
        names.push_back("d2W_dab_dtheta");
    if (out.d2W_dab_dabarbbar)
        names.push_back("d2W_dab_dabarbbar");
    const nb::object result =
        call_eval(instance, nb::make_tuple(rows(in.ab), rows(in.ab_bar), vector(in.thickness), rows(in.theta)), names);
    if (out.W)
        read_vector(instance, result, "W", count, *out.W);
    if (out.dW_dab)
        read_matrices(instance, result, "dW_dab", {count, 6}, 6, 1, *out.dW_dab);
    if (out.d2W_dab2)
        read_matrices(instance, result, "d2W_dab2", {count, 6, 6}, 6, 6, *out.d2W_dab2);
    if (out.d2W_dab_dtheta)
        read_matrices(instance, result, "d2W_dab_dtheta", {count, 6, k}, 6, num_parameters_, *out.d2W_dab_dtheta);
    if (out.d2W_dab_dabarbbar)
        read_matrices(instance, result, "d2W_dab_dabarbbar", {count, 6, 6}, 6, 6, *out.d2W_dab_dabarbbar);
}

namespace {
nb::object write_matrices(const Eigen::Ref<const Eigen::MatrixXd>& packed, std::initializer_list<size_t> shape,
                          int row_blocks, int cols, bool tensor_columns = false) {
    const int rows = row_blocks == 3 ? 9 : row_blocks;
    return new_array(shape, [&](double* data) {
        for (Eigen::Index n = 0; n < packed.cols(); ++n)
            for (int r = 0; r < rows; ++r)
                for (int c = 0; c < cols; ++c) {
                    const int target_r = row_blocks == 3 ? 3 * (r % 3) + r / 3 : r;
                    const int target_c = tensor_columns ? 3 * (c % 3) + c / 3 : c;
                    data[(n * rows + target_r) * cols + target_c] = packed(r + rows * c, n);
                }
    });
}
size_t input_count(nb::handle value) {
    const Input array =
        nb::cast<Input>(nb::module_::import_("numpy").attr("ascontiguousarray")(value, "dtype"_a = "float64"));
    if (array.ndim() < 1)
        throw nb::value_error("material inputs must have a batch axis");
    return array.shape(0);
}
VolumetricMaterialBatch volume_inputs(nb::handle owner, nb::object F, nb::object Fp, nb::object theta, int k) {
    const auto n = input_count(F);
    nb::dict args;
    args["F"] = F;
    args["Fp"] = Fp;
    args["theta"] = theta;
    VolumetricMaterialBatch in;
    read_matrices(owner, args, "F", {n, 3, 3}, 3, 1, in.F);
    read_matrices(owner, args, "Fp", {n, 3, 3}, 3, 1, in.Fp);
    const Input params = read_output(owner, args, "theta", {n, static_cast<size_t>(k)});
    in.theta = Eigen::Map<const Eigen::MatrixXd>(params.data(), k, static_cast<Eigen::Index>(n));
    return in;
}
nb::dict volume_eval(const VolumetricBatchMaterial& material, nb::object F, nb::object Fp, nb::object theta,
                     const std::vector<std::string>& outputs) {
    auto owner = nb::cast(&material, nb::rv_policy::reference);
    const auto in = volume_inputs(owner, F, Fp, theta, material.num_parameters());
    const auto n = static_cast<size_t>(in.size()), k = static_cast<size_t>(material.num_parameters());
    Vec W;
    Eigen::Matrix<double, 9, Eigen::Dynamic> P;
    Eigen::Matrix<double, 81, Eigen::Dynamic> A, plastic;
    Eigen::MatrixXd params;
    VolumetricMaterialBatchOutputs out;
    for (const auto& name : outputs) {
        if (name == "W")
            out.W = &W;
        else if (name == "dW_dF")
            out.dW_dF = &P;
        else if (name == "d2W_dF2")
            out.d2W_dF2 = &A;
        else if (name == "d2W_dF_dtheta")
            out.d2W_dF_dtheta = &params;
        else if (name == "d2W_dF_dFp")
            out.d2W_dF_dFp = &plastic;
        else
            throw nb::value_error(("Unknown material output: " + name).c_str());
    }
    {
        nb::gil_scoped_release release;
        material.eval(in, out);
    }
    detail::check_volumetric_batch_outputs(out, in.size(), material.num_parameters());
    nb::dict result;
    if (out.W)
        result["W"] = vector(W);
    if (out.dW_dF)
        result["dW_dF"] = matrices(P);
    if (out.d2W_dF2)
        result["d2W_dF2"] = write_matrices(A, {n, 3, 3, 3, 3}, 3, 9, true);
    if (out.d2W_dF_dtheta)
        result["d2W_dF_dtheta"] = write_matrices(params, {n, 3, 3, k}, 3, static_cast<int>(k));
    if (out.d2W_dF_dFp)
        result["d2W_dF_dFp"] = write_matrices(plastic, {n, 3, 3, 3, 3}, 3, 9, true);
    return result;
}
nb::dict shell_eval(const ShellBatchMaterial& material, nb::object ab, nb::object ab_bar, nb::object thickness,
                    nb::object theta, const std::vector<std::string>& outputs) {
    auto owner = nb::cast(&material, nb::rv_policy::reference);
    const auto n = input_count(ab), k = static_cast<size_t>(material.num_parameters());
    nb::dict args;
    args["ab"] = ab;
    args["ab_bar"] = ab_bar;
    args["thickness"] = thickness;
    args["theta"] = theta;
    ShellMaterialBatch in;
    read_matrices(owner, args, "ab", {n, 6}, 6, 1, in.ab);
    read_matrices(owner, args, "ab_bar", {n, 6}, 6, 1, in.ab_bar);
    read_vector(owner, args, "thickness", n, in.thickness);
    const auto params = read_output(owner, args, "theta", {n, k});
    in.theta =
        Eigen::Map<const Eigen::MatrixXd>(params.data(), static_cast<Eigen::Index>(k), static_cast<Eigen::Index>(n));
    Vec W;
    Eigen::Matrix<double, 6, Eigen::Dynamic> P;
    Eigen::Matrix<double, 36, Eigen::Dynamic> A, natural;
    Eigen::MatrixXd mixed;
    ShellMaterialBatchOutputs out;
    for (const auto& name : outputs) {
        if (name == "W")
            out.W = &W;
        else if (name == "dW_dab")
            out.dW_dab = &P;
        else if (name == "d2W_dab2")
            out.d2W_dab2 = &A;
        else if (name == "d2W_dab_dtheta")
            out.d2W_dab_dtheta = &mixed;
        else if (name == "d2W_dab_dabarbbar")
            out.d2W_dab_dabarbbar = &natural;
        else
            throw nb::value_error(("Unknown material output: " + name).c_str());
    }
    {
        nb::gil_scoped_release release;
        material.eval(in, out);
    }
    detail::check_shell_batch_outputs(out, in.size(), material.num_parameters());
    nb::dict result;
    if (out.W)
        result["W"] = vector(W);
    if (out.dW_dab)
        result["dW_dab"] = rows(P);
    if (out.d2W_dab2)
        result["d2W_dab2"] = write_matrices(A, {n, 6, 6}, 6, 6);
    if (out.d2W_dab_dtheta)
        result["d2W_dab_dtheta"] = write_matrices(mixed, {n, 6, k}, 6, static_cast<int>(k));
    if (out.d2W_dab_dabarbbar)
        result["d2W_dab_dabarbbar"] = write_matrices(natural, {n, 6, 6}, 6, 6);
    return result;
}
nb::object volume_feasible_step(const VolumetricBatchMaterial& material, nb::object F, nb::object Fp, nb::object theta,
                                nb::object dF, bool use_default) {
    auto owner = nb::cast(&material, nb::rv_policy::reference);
    auto in = volume_inputs(owner, F, Fp, theta, material.num_parameters());
    nb::dict args;
    args["dF"] = dF;
    Eigen::Matrix<double, 9, Eigen::Dynamic> delta;
    read_matrices(owner, args, "dF", {static_cast<size_t>(in.size()), 3, 3}, 3, 1, delta);
    Vec steps;
    {
        nb::gil_scoped_release release;
        steps = use_default ? default_feasible_step(in, delta) : material.eval_feasible_step(in, delta);
    }
    if (steps.size() != in.size() || !steps.allFinite() || (steps.array() < 0).any() || (steps.array() > 1).any())
        throw nb::value_error("material returned invalid feasible steps");
    return vector(steps);
}
} // namespace
void bind_batch_materials(nb::module_& m) {
    nb::class_<VolumetricBatchMaterial>(m, "VolumetricBatchMaterial")
        .def_prop_ro("num_parameters", &VolumetricBatchMaterial::num_parameters)
        .def("eval", &volume_eval, "F"_a, "Fp"_a, "theta"_a, nb::kw_only(), "outputs"_a)
        .def(
            "eval_feasible_step",
            [](const VolumetricBatchMaterial& material, nb::object F, nb::object Fp, nb::object theta, nb::object dF) {
                return volume_feasible_step(material, F, Fp, theta, dF, false);
            },
            "F"_a, "Fp"_a, "theta"_a, "dF"_a);
    nb::class_<ShellBatchMaterial>(m, "ShellBatchMaterial")
        .def_prop_ro("num_parameters", &ShellBatchMaterial::num_parameters)
        .def("eval", &shell_eval, "ab"_a, "ab_bar"_a, "thickness"_a, "theta"_a, nb::kw_only(), "outputs"_a);
    nb::class_<PyVolumetricMaterial, VolumetricBatchMaterial>(m, "PyVolumetricMaterial")
        .def("__init__",
             [](nb::pointer_and_handle<PyVolumetricMaterial> self) { new (self.p) PyVolumetricMaterial(self.h); })
        // Bypass virtual dispatch so an override can call super() without entering itself again.
        .def(
            "eval_feasible_step",
            [](const PyVolumetricMaterial& material, nb::object F, nb::object Fp, nb::object theta, nb::object dF) {
                return volume_feasible_step(material, F, Fp, theta, dF, true);
            },
            "F"_a, "Fp"_a, "theta"_a, "dF"_a);
    nb::class_<PyShellMaterial, ShellBatchMaterial>(m, "PyShellMaterial")
        .def("__init__", [](nb::pointer_and_handle<PyShellMaterial> self) { new (self.p) PyShellMaterial(self.h); });
    using Volume = PointwiseVolumetricAdapter<NeoHookean>;
    using Shell = PointwiseShellAdapter<KoiterStVK>;
    nb::class_<Volume, VolumetricBatchMaterial>(m, "PointwiseVolumetricAdapter[NeoHookean]").def(nb::init<>());
    nb::class_<Shell, ShellBatchMaterial>(m, "PointwiseShellAdapter[KoiterStVK]").def(nb::init<>());
    bindings::register_instance(m, "PointwiseVolumetricAdapter", nb::make_tuple(nb::type<NeoHookean>()),
                                nb::type<Volume>());
    bindings::register_instance(m, "PointwiseShellAdapter", nb::make_tuple(nb::type<KoiterStVK>()), nb::type<Shell>());
}

} // namespace litepgo::fem::binding
