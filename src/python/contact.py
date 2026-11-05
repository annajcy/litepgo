"""IPC self-contact, optional IPC Toolkit, and a one-sided plane penalty.

Energies act on vertex-major xyz positions, with unit stiffness. Call
update_pattern(x) before Hessian assembly. Compose with AffineMappedEnergy and
SurfaceKinematics for displacement DOFs. Native ACCD uses floating-point arithmetic;
it does not provide formal floating-point robustness guarantees. Native energies
require an explicit broad_phase=SpatialHash() and retain shared
ownership of that backend. Custom backends are implemented in C++, not Python.
Both options types default to fixed reference-area weights; set
use_area_weighting=False for the original unweighted candidate sum.
ContactPairWeights supplies fixed primitive groups and a symmetric multiplier
table, applied before feature merging. Zero excludes a candidate from the active
set and CCD; whole-surface initial-intersection validation remains independent.
contact_snapshot(x) copies every merged active record without updating the pattern;
its arrays are read-only and independent of subsequent queries.
"""
from litepgo import _core

BroadPhase = _core.contact.BroadPhase
SpatialHash = _core.contact.SpatialHash
PlanePenaltyEnergy = _core.contact.PlanePenaltyEnergy
IPCContactOptions = _core.contact.IPCContactOptions
IPCContactEnergy = _core.contact.IPCContactEnergy
ContactSnapshot = _core.contact.ContactSnapshot
ContactPairWeights = _core.contact.ContactPairWeights
contact_edges = _core.contact.contact_edges
__all__ = ["BroadPhase", "SpatialHash", "PlanePenaltyEnergy", "IPCContactOptions", "IPCContactEnergy", "ContactSnapshot"]
__all__ += ["ContactPairWeights", "contact_edges"]

if hasattr(_core.contact, "IPCTKContactEnergy"):
    IPCTKContactOptions = _core.contact.IPCTKContactOptions
    IPCTKContactEnergy = _core.contact.IPCTKContactEnergy
    __all__ += ["IPCTKContactOptions", "IPCTKContactEnergy"]
