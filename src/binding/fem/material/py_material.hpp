#pragma once
#include "litepgo/fem/material/shell/shell_material_batch.hpp"
#include "litepgo/fem/material/volumetric/volumetric_material_batch.hpp"
#include <nanobind/nanobind.h>

namespace litepgo::fem::binding {

// The native wrapper borrows its owning Python instance. Nanobind's shared_ptr caster
// keeps that instance alive while a FEM object holds the material.
class PyVolumetricMaterial : public VolumetricBatchMaterial {
  public:
    explicit PyVolumetricMaterial(nanobind::handle instance);
    int num_parameters() const override {
        return num_parameters_;
    }
    void eval(const VolumetricMaterialBatch& inputs, VolumetricMaterialBatchOutputs& outputs) const override;
    Vec eval_feasible_step(const VolumetricMaterialBatch& inputs,
                           const Eigen::Matrix<double, 9, Eigen::Dynamic>& dF) const override;

  private:
    nanobind::handle instance_;
    int num_parameters_;
    bool has_feasible_step_;
};

// Adapts a Python subclass of litepgo.fem.PyShellMaterial to ShellBatchMaterial.
// eval calls instance.eval(ab, ab_bar, thickness, theta, outputs=(...)) with packed (N,6) forms.
class PyShellMaterial : public ShellBatchMaterial {
  public:
    explicit PyShellMaterial(nanobind::handle instance);
    int num_parameters() const override {
        return num_parameters_;
    }
    void eval(const ShellMaterialBatch& inputs, ShellMaterialBatchOutputs& outputs) const override;

  private:
    nanobind::handle instance_;
    int num_parameters_;
};

void bind_batch_materials(nanobind::module_& m);
} // namespace litepgo::fem::binding
