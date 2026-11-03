# Collision safety and diagnostics

Contact evaluation and step bounds consume packed xyz surface-position vectors.
Broad-phase candidates must cover every permitted overlap; CCD then bounds
motion along a search direction. The [barrier page](barrier.md) defines the
activation width and minimum separation.

## Broad phase and collision-safe steps

`contact.SpatialHash()` selects an automatic cell width from primitive AABB
extents; a finite positive explicit width is also accepted. Static/swept bounds
are inflated and outward-rounded. Native C++ custom backends implement
`BroadPhase::detect(query) const` and must return every allowed overlapping pair.
Extra pairs, duplicates, and arbitrary ordering are accepted and filtered;
missing pairs violate the collision-safety contract. Backends must support
concurrent queries without retaining query storage. Python can select bound
backends but cannot override one with a Python implementation.

The hash validates bounds and limits cell records to $2^{20}$. Long overlapping
boxes and dense cells can still have quadratic pair work; no pair-work-triggered
fallback is provided. Box construction/active-set narrowphase contain serial work.
Local contact, projection, assembly, and surface CCD use TBB. Native assembly
orders contributions deterministically across thread counts.

`max_feasible_step` uses swept candidates and additive conservative advancement.
`ccd_tolerance` is a **time-increment** stopping tolerance, and
`ccd_max_iterations` limits work per pair. On exhaustion/stalling the algorithm
returns the last conservative bound, not an unchecked full step. Plane steps use
an analytic signed-distance bound with a 0.8 safety factor. Plane normals are
normalized and their positive side is feasible.

## Validity, diagnostics, and optional backend

The surface must be nonempty, have nondegenerate unique triangles, and use every
vertex. Start from a feasible nonintersecting state. `eval` rejects active
separations at/below `min_distance` and invalid plane positions, but is not a full
intersection query; the step method additionally checks initial edge–face
intersections. Shared-vertex primitives are excluded. Degenerate trial triangles
are rejected.

`num_active_pairs(p)` counts merged active stencils, including vertex–plane
terms. `min_active_distance(p)` returns the smallest distance $d$ among them,
or positive infinity if none are active. It does not subtract `min_distance`.
Here $d_{\min}$ denotes `options.min_distance` and $\hat d$ denotes `options.dhat`.
The feasible active interval is $d_{\min}<d<d_{\min}+\hat d$, equivalently
$0<d-d_{\min}<\hat d$. Use mapped surface positions rather than volume
displacement for these methods.

`IPCTKContactEnergy` / `IPCTKContactOptions` are separate optional bindings,
enabled with `LITEPGO_WITH_IPC_TOOLKIT=ON` and linked in C++ with
`litepgo::contact_ipctk`. They use the Toolkit's CCD behavior rather
than native ACCD. Native derivatives contain a vendored subset of analytical
IPC routines with their own retained license; native feature selection remains
independent. Floating-point predicates/ACCD have numerical limits at extreme
scales, grazing contact, and degeneracy. No exact-arithmetic robustness, friction,
or adaptive barrier-stiffness scheme is provided.
