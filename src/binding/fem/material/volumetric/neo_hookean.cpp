#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "bindings.hpp"

#include <nanobind/eigen/dense.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::fem::binding {
namespace {
// Volumetric materials are tags with their pointwise density and derivatives.
template <class M> void bind_neo_hookean(nb::module_& m, const char* name) {
    nb::class_<M> cls(m, name);
    cls.attr("num_parameters") = M::num_parameters;
#define LITEPGO_VOLUMETRIC(fn)                                                                                         \
    cls.def_static(                                                                                                    \
        #fn,                                                                                                           \
        [](const Eigen::Matrix3d& F, const Eigen::Matrix3d& Fp, const Eigen::VectorXd& theta) {                        \
            return M::fn(VolumetricMaterialInputs{F, Fp, theta});                                                      \
        },                                                                                                             \
        "F"_a, "Fp"_a, "theta"_a)
    LITEPGO_VOLUMETRIC(eval_W);
    LITEPGO_VOLUMETRIC(eval_dW_dF);
    LITEPGO_VOLUMETRIC(eval_d2W_dF2);
    LITEPGO_VOLUMETRIC(eval_d2W_dF_dtheta);
    LITEPGO_VOLUMETRIC(eval_d2W_dF_dFp);
#undef LITEPGO_VOLUMETRIC
    cls.def_static(
        "eval_feasible_step",
        [](const Eigen::Matrix3d& F, const Eigen::Matrix3d& Fp, const Eigen::VectorXd& theta, const Eigen::Matrix3d& dF,
           const Eigen::Matrix3d& dFp, const Eigen::VectorXd& dtheta) {
            return M::eval_feasible_step(VolumetricMaterialInputs{F, Fp, theta},
                                         VolumetricMaterialInputs{dF, dFp, dtheta});
        },
        "F"_a, "Fp"_a, "theta"_a, "dF"_a, "dFp"_a, "dtheta"_a,
        "A conservative alpha in [0, 1] keeping inputs + t * increments valid for every t in [0, alpha].");
}
} // namespace

void bind_fem_material_volumetric_neo_hookean(nb::module_& m) {
    bind_neo_hookean<NeoHookean>(m, "NeoHookean");
}
} // namespace litepgo::fem::binding
