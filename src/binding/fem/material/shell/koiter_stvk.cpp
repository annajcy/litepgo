#include "litepgo/fem/material/shell/koiter_stvk.hpp"
#include "bindings.hpp"

#include <nanobind/eigen/dense.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::fem::binding {
namespace {
template <class M> void bind_koiter_stvk(nb::module_& m, const char* name) {
    nb::class_<M> cls(m, name);
    cls.attr("num_parameters") = M::num_parameters;
#define LITEPGO_SHELL(fn)                                                                                              \
    cls.def_static(                                                                                                    \
        #fn,                                                                                                           \
        [](const Eigen::Matrix2d& a, const Eigen::Matrix2d& b, const Eigen::Matrix2d& a_bar,                           \
           const Eigen::Matrix2d& b_bar, double thickness,                                                             \
           const Eigen::VectorXd& theta) { return M::fn(ShellMaterialInputs{a, b, a_bar, b_bar, thickness, theta}); }, \
        "a"_a, "b"_a, "a_bar"_a, "b_bar"_a, "thickness"_a, "theta"_a)
    LITEPGO_SHELL(eval_W);
    LITEPGO_SHELL(eval_dW_dab);
    LITEPGO_SHELL(eval_d2W_dab2);
    LITEPGO_SHELL(eval_d2W_dab_dtheta);
    LITEPGO_SHELL(eval_d2W_dab_dabarbbar);
#undef LITEPGO_SHELL
}
} // namespace

void bind_fem_material_shell_koiter_stvk(nb::module_& m) {
    bind_koiter_stvk<KoiterStVK>(m, "KoiterStVK");
}
} // namespace litepgo::fem::binding
