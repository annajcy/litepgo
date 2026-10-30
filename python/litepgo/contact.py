"""Native IPC surface contact and optional IPC Toolkit backend.

Energies act on vertex-major xyz positions, with unit stiffness. Call
update_pattern(x) before Hessian assembly. Compose with AffineMappedEnergy and
SurfaceDofMap for displacement DOFs. Native ACCD uses floating-point arithmetic;
it does not provide formal floating-point robustness guarantees.
"""
from litepgo import _core

Plane = _core.contact.Plane
IPCContactOptions = _core.contact.IPCContactOptions
IPCContactEnergy = _core.contact.IPCContactEnergy
__all__ = ["Plane", "IPCContactOptions", "IPCContactEnergy"]

if hasattr(_core.contact, "IPCTKContactEnergy"):
    IPCTKContactOptions = _core.contact.IPCTKContactOptions
    IPCTKContactEnergy = _core.contact.IPCTKContactEnergy
    __all__ += ["IPCTKContactOptions", "IPCTKContactEnergy"]
