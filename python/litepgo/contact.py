"""IPC barrier contact on IPC Toolkit, mirroring cpp/litepgo/contact.

Requires a build with LITEPGO_WITH_IPC=ON; build_info()["ipc_toolkit_version"] is
empty otherwise. BarrierContactEnergy is an energy.Energy over the surface vertices'
positions, with stiffness 1: give it its stiffness as a weight in a WeightedEnergySum.
Its Hessian pattern follows the active contact pairs, so call update_pattern(x)
before assembling a Hessian at x. To drive it with displacement DOFs u, wrap it as
energy.AffineMappedEnergy(barrier, m.displacement_map, m.rest_positions.ravel())
for a dof.SurfaceDofMap m.
"""

from litepgo import _core

if not hasattr(_core, "contact"):
    raise ImportError("litepgo.contact requires a build with LITEPGO_WITH_IPC=ON")

_native = _core.contact

Plane = _native.Plane
BarrierContactOptions = _native.BarrierContactOptions
BarrierContactEnergy = _native.BarrierContactEnergy

__all__ = ["Plane", "BarrierContactOptions", "BarrierContactEnergy"]
