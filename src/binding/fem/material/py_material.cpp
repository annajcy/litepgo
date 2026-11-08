#include "fem/material/py_material.hpp"
#include "litepgo/fem/kinematics/positive_det_feasible_step.hpp"
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

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
Input read_output(nb::handle cls, nb::handle result, const char* name, std::initializer_list<size_t> shape) {
    const std::string owner = nb::cast<std::string>(nb::getattr(cls, "__name__")) +
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

nb::object call_eval(nb::handle cls, nb::tuple arguments, const std::vector<const char*>& names) {
    nb::list requested;
    for (const char* name : names)
        requested.append(nb::str(name));
    nb::object result = cls.attr("eval")(*arguments, "outputs"_a = nb::tuple(requested));
    if (!nb::isinstance<nb::dict>(result))
        throw nb::type_error(
            (nb::cast<std::string>(nb::getattr(cls, "__name__")) + ".eval must return a dict").c_str());
    return result;
}

int read_num_parameters(nb::handle cls) {
    nb::object value = nb::getattr(cls, "num_parameters", nb::none());
    if (!nb::isinstance<nb::int_>(value) || nb::isinstance<nb::bool_>(value) || nb::cast<long long>(value) < 0 ||
        nb::cast<long long>(value) > 1 << 20)
        throw nb::type_error("material class must define a non-negative integer num_parameters");
    if (!PyCallable_Check(nb::getattr(cls, "eval", nb::none()).ptr()))
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

void read_vector(nb::handle cls, nb::handle result, const char* name, size_t count, Vec& out) {
    const Input values = read_output(cls, result, name, {count});
    out = Eigen::Map<const Vec>(values.data(), static_cast<Eigen::Index>(count));
}

// [n, a..., b...] in C order to column n holding a column-major (rows x cols) matrix; the
// row index is (i,j) with i+3*j for 3x3 blocks, or r for packed vectors.
template <class Out>
void read_matrices(nb::handle cls, nb::handle result, const char* name, std::initializer_list<size_t> shape,
                   int row_blocks, int cols, Out& out) {
    const Input values = read_output(cls, result, name, shape);
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
} // namespace

void check_material_owner(nb::handle instance, nb::handle material) {
    if (!instance.is_valid() || !nb::getattr(instance.type(), "material", nb::none()).is(material))
        throw nb::type_error("Python-material FEM classes must be created through litepgo.fem template indexing, "
                             "e.g. VolumetricFemEnergy[MyMaterial, Basis, Quadrature](...)");
}

PyVolumetricMaterial::PyVolumetricMaterial(nb::handle cls)
    : cls_(cls), num_parameters_(read_num_parameters(cls)),
      has_feasible_step_(!nb::getattr(cls, "eval_feasible_step", nb::none()).is_none()) {}

void PyVolumetricMaterial::eval(const VolumetricMaterialBatch& in, VolumetricMaterialBatchOutputs& out) const {
    nb::gil_scoped_acquire gil;
    const nb::handle cls = cls_;
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
    const nb::object result = call_eval(cls, nb::make_tuple(matrices(in.F), matrices(in.Fp), rows(in.theta)), names);
    if (out.W)
        read_vector(cls, result, "W", count, *out.W);
    if (out.dW_dF)
        read_matrices(cls, result, "dW_dF", {count, 3, 3}, 3, 1, *out.dW_dF);
    if (out.d2W_dF2) {
        read_matrices(cls, result, "d2W_dF2", {count, 3, 3, 3, 3}, 3, 9, *out.d2W_dF2);
        permute_matrix_columns(*out.d2W_dF2);
    }
    if (out.d2W_dF_dtheta)
        read_matrices(cls, result, "d2W_dF_dtheta", {count, 3, 3, k}, 3, num_parameters_, *out.d2W_dF_dtheta);
    if (out.d2W_dF_dFp) {
        read_matrices(cls, result, "d2W_dF_dFp", {count, 3, 3, 3, 3}, 3, 9, *out.d2W_dF_dFp);
        permute_matrix_columns(*out.d2W_dF_dFp);
    }
}

Vec PyVolumetricMaterial::eval_feasible_step(const VolumetricMaterialBatch& in,
                                             const Eigen::Matrix<double, 9, Eigen::Dynamic>& dF) const {
    const Eigen::Index count = in.size();
    Vec steps(count);
    if (!has_feasible_step_) {
        for (Eigen::Index n = 0; n < count; ++n)
            steps[n] = eval_positive_det_feasible_step(in.F.col(n).reshaped(3, 3), dF.col(n).reshaped(3, 3));
        return steps;
    }
    nb::gil_scoped_acquire gil;
    const nb::handle cls = cls_;
    nb::object result = cls.attr("eval_feasible_step")(matrices(in.F), matrices(in.Fp), rows(in.theta), matrices(dF));
    nb::dict wrapped;
    wrapped["steps"] = result;
    read_vector(cls, wrapped, "steps", static_cast<size_t>(count), steps);
    return steps;
}

PyShellMaterial::PyShellMaterial(nb::handle cls) : cls_(cls), num_parameters_(read_num_parameters(cls)) {}

void PyShellMaterial::eval(const ShellMaterialBatch& in, ShellMaterialBatchOutputs& out) const {
    nb::gil_scoped_acquire gil;
    const nb::handle cls = cls_;
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
        call_eval(cls, nb::make_tuple(rows(in.ab), rows(in.ab_bar), vector(in.thickness), rows(in.theta)), names);
    if (out.W)
        read_vector(cls, result, "W", count, *out.W);
    if (out.dW_dab)
        read_matrices(cls, result, "dW_dab", {count, 6}, 6, 1, *out.dW_dab);
    if (out.d2W_dab2)
        read_matrices(cls, result, "d2W_dab2", {count, 6, 6}, 6, 6, *out.d2W_dab2);
    if (out.d2W_dab_dtheta)
        read_matrices(cls, result, "d2W_dab_dtheta", {count, 6, k}, 6, num_parameters_, *out.d2W_dab_dtheta);
    if (out.d2W_dab_dabarbbar)
        read_matrices(cls, result, "d2W_dab_dabarbbar", {count, 6, 6}, 6, 6, *out.d2W_dab_dabarbbar);
}

} // namespace litepgo::fem::binding
