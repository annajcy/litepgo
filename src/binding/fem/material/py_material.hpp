#pragma once
#include "litepgo/fem/material/shell/shell_material_batch.hpp"
#include "litepgo/fem/material/volumetric/volumetric_material_batch.hpp"
#include <nanobind/nanobind.h>

namespace litepgo::fem::binding {

// Material classes are borrowed: the owning Python instance's type keeps them alive through its
// `material` attribute (set by litepgo.fem), so no reference cycle passes through C++.
// Raises TypeError unless type(instance).material is material.
void check_material_owner(nanobind::handle instance, nanobind::handle material);

// Adapts a Python subclass of litepgo.fem.PyVolumetricMaterial to VolumetricBatchMaterial.
// Construct with the GIL held; the class must outlive this adapter. eval calls cls.eval(F, Fp, theta, outputs=(...))
// once per batch with (N,3,3) and (N,k) float64 arrays and copies the requested entries of the returned dict. Without
// cls.eval_feasible_step, the bound keeps det(F) > 0.
class PyVolumetricMaterial {
  public:
    explicit PyVolumetricMaterial(nanobind::handle cls);
    int num_parameters() const {
        return num_parameters_;
    }
    void eval(const VolumetricMaterialBatch& inputs, VolumetricMaterialBatchOutputs& outputs) const;
    Vec eval_feasible_step(const VolumetricMaterialBatch& inputs,
                           const Eigen::Matrix<double, 9, Eigen::Dynamic>& dF) const;

  private:
    nanobind::handle cls_;
    int num_parameters_;
    bool has_feasible_step_;
};

// Adapts a Python subclass of litepgo.fem.PyShellMaterial to ShellBatchMaterial.
// eval calls cls.eval(ab, ab_bar, thickness, theta, outputs=(...)) with packed (N,6) forms.
class PyShellMaterial {
  public:
    explicit PyShellMaterial(nanobind::handle cls);
    int num_parameters() const {
        return num_parameters_;
    }
    void eval(const ShellMaterialBatch& inputs, ShellMaterialBatchOutputs& outputs) const;

  private:
    nanobind::handle cls_;
    int num_parameters_;
};

static_assert(VolumetricBatchMaterial<PyVolumetricMaterial>);
static_assert(ShellBatchMaterial<PyShellMaterial>);

// Template keys: native materials use their bound type; Python materials a name that
// litepgo.fem replaces with its PyVolumetricMaterial/PyShellMaterial base class.
template <class M> nanobind::object material_key() {
    if constexpr (std::same_as<M, PyVolumetricMaterial>)
        return nanobind::str("PyVolumetricMaterial");
    else if constexpr (std::same_as<M, PyShellMaterial>)
        return nanobind::str("PyShellMaterial");
    else
        return nanobind::borrow(nanobind::type<M>());
}
// The adapter for the C++ object being initialized, after checking that its Python owner keeps material alive.
template <class M, class T> M owned_material(const nanobind::pointer_and_handle<T>& self, nanobind::handle material) {
    check_material_owner(self.h, material);
    return M(material);
}
template <class M>
constexpr bool is_py_material = std::same_as<M, PyVolumetricMaterial> || std::same_as<M, PyShellMaterial>;

} // namespace litepgo::fem::binding
