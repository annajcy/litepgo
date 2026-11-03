"""IPC self-contact, optional IPC Toolkit, and a one-sided plane penalty.

Energies act on vertex-major xyz positions, with unit stiffness. Call
update_pattern(x) before Hessian assembly. Compose with AffineMappedEnergy and
SurfaceKinematics for displacement DOFs. Native ACCD uses floating-point arithmetic;
it does not provide formal floating-point robustness guarantees. Native energies
require an explicit broad_phase=SpatialHash() and retain shared
ownership of that backend. Custom backends are implemented in C++, not Python.
"""
from litepgo import _core

BroadPhase = _core.contact.BroadPhase
SpatialHash = _core.contact.SpatialHash
PlanePenaltyEnergy = _core.contact.PlanePenaltyEnergy
IPCContactOptions = _core.contact.IPCContactOptions
IPCContactEnergy = _core.contact.IPCContactEnergy
__all__ = ["BroadPhase", "SpatialHash", "PlanePenaltyEnergy", "IPCContactOptions", "IPCContactEnergy"]

if hasattr(_core.contact, "IPCTKContactEnergy"):
    IPCTKContactOptions = _core.contact.IPCTKContactOptions
    IPCTKContactEnergy = _core.contact.IPCTKContactEnergy
    __all__ += ["IPCTKContactOptions", "IPCTKContactEnergy"]
