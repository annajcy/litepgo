"""Native IPC surface contact and optional IPC Toolkit backend.

Energies act on vertex-major xyz positions, with unit stiffness. Call
update_pattern(x) before Hessian assembly. Compose with AffineMappedEnergy and
SurfaceDofMap for displacement DOFs. Native ACCD uses floating-point arithmetic;
it does not provide formal floating-point robustness guarantees. Native energies
require an explicit broad_phase=SweepAndPrune() or SpatialHash() and retain shared
ownership of that backend. Custom backends are implemented in C++, not Python.
"""
from litepgo import _core

BroadPhase = _core.contact.BroadPhase
SweepAndPrune = _core.contact.SweepAndPrune
SpatialHash = _core.contact.SpatialHash
Plane = _core.contact.Plane
IPCContactOptions = _core.contact.IPCContactOptions
IPCContactEnergy = _core.contact.IPCContactEnergy
__all__ = ["BroadPhase", "SweepAndPrune", "SpatialHash", "Plane", "IPCContactOptions", "IPCContactEnergy"]

if hasattr(_core.contact, "IPCTKContactEnergy"):
    IPCTKContactOptions = _core.contact.IPCTKContactOptions
    IPCTKContactEnergy = _core.contact.IPCTKContactEnergy
    __all__ += ["IPCTKContactOptions", "IPCTKContactEnergy"]
