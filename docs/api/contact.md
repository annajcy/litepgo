# Contact

`litepgo/contact/ipc_contact_energy.hpp` provides independent native `contact::IPCContactEnergy` and
`contact::IPCContactOptions`. `litepgo/contact/ipctk_contact_energy.hpp` provides the retained toolkit backend,
`contact::IPCTKContactEnergy` and `contact::IPCTKContactOptions`, linked with `litepgo::ipctk`.
Both use `contact::Plane`, and are `energy::Energy` implementations over vertex-major xyz surface positions
with unit stiffness. Set the stiffness as a weight in `WeightedEnergySum`. Call `update_pattern(x)` before
Hessian assembly; the Newton solver handles the dynamic sparsity pattern automatically.

Native contact implements the original squared-distance log barrier
`b(s,h) = -(s-h)^2 log(s/h)` for `0 < s < h`, zero for `s >= h`, with
`s = distance² - min_distance²` and `h = dhat * (dhat + 2*min_distance)`.
It sums nonincident vertex-triangle and unordered edge-edge candidates, with closest-feature
point-point, point-edge, point-triangle and edge-edge distances. Repeated closest-feature stencils are
merged **with their multiplicities preserved**: this is an assembly optimization, not a change from the
original candidate sum. Active-pair diagnostics count merged stencils. Edge pairs needing mollification
retain all four vertices, using `m = z*(2-z)` for `z < 1`, otherwise one, where
`z = |e0 × e1|² / (1e-3 |e0_rest|² |e1_rest|²)`. Derivatives include the distance and mollifier;
projected Hessians clamp each local term's eigenvalues. Assembly order is fixed and deterministic.
This is the unweighted original barrier, not the area-weighted/physical barrier or friction model.

The native backend uses spatial-hash AABBs, including swept boxes for linear motion, and additive
conservative advancement (ACCD). `IPCContactOptions.ccd_tolerance` is a **time increment** stopping
tolerance: when another safe increment is smaller, return the last conservative bound.
`ccd_max_iterations` bounds advancement per pair, also returning the last bound on exhaustion.
The first positive conservative increment is allowed even below the tolerance; stalled arithmetic
returns zero. Neither option switches off CCD. `IPCTKContactOptions` instead passes its tolerance and
iteration budget to Tight-Inclusion CCD. Planes use analytic signed-distance steps with a 0.8 safety factor.

Native input requires nonempty, nondegenerate triangles, unique faces, and no unused vertices.
Options, finite positions, normalized plane normals and representable geometric quantities are validated.
`eval` rejects active distances at or below `min_distance` and vertices behind planes; it is not a full
intersection query. `max_feasible_step` additionally checks initial edge-face intersections. Start solves
from feasible states. Shared-vertex primitives are excluded. Coplanar overlaps are detected through
zero closest-feature distances. Degenerate evaluated triangles are rejected.

Native distance gradients/Hessians use a minimal vendored subset of IPC Toolkit 1.6.1
analytical routines (MIT; see `cpp/litepgo/contact/detail/ipc_derivatives/README.md`
and its original LICENSE). Feature selection and scalar distances remain native.
Barrier and mollifier derivatives use explicit chain/product rules, including mixed
terms; gradient-only evaluation does not compute Hessians. This does not link the
native library to IPC Toolkit and does not change the optional IPCTK backend.

Native predicates and ACCD use double precision, without a formal floating-point robustness guarantee;
extreme scales, grazing contact and nearly degenerate geometry remain numerical limitations.
Closest-feature transitions are generally C1, not C2; exact Hessians are those of the selected branch.
Native contact accepts a shared, read-only broadphase backend as its required third argument:

```cpp
#include <litepgo/contact/broad_phase/spatial_hash.hpp>
auto broad_phase = std::make_shared<litepgo::contact::SpatialHash>();
litepgo::contact::IPCContactEnergy contact(surface, options, broad_phase);
```

```python
from litepgo.contact import IPCContactEnergy, SpatialHash
broad_phase = SpatialHash()
contact = IPCContactEnergy(surface, options, broad_phase=broad_phase)
```

The native constructor is `(surface, options, broad_phase, planes = {})`.
The backend is required: omission, `nullptr` and `None` are rejected. Explicitly pass `SpatialHash`; move any plane list to the fourth argument. Energies retain shared ownership, so one backend
can serve multiple energies and outlive the caller's reference. IPCTK's constructor is unchanged.
Python exposes the built-in base/derived types for selection, not Python-defined backend overrides.

C++ backends derive from `BroadPhase` and implement `detect(const BroadPhaseQuery&) const`.
The query contains already inflated, outward-rounded static or swept `AABB`s and a
`PairKindMask` for VF, EE and optional EF pairs. Return `BoxPair` indices into the input
array, including every allowed overlapping pair. Missing pairs violate the collision-safety
contract. Conservative extra pairs, arbitrary order and duplicates are accepted; native
contact validates indices, filters kinds/AABB overlap/topology, and sorts/uniques primitive
pairs before contact feature multiplicities are formed. Backend calls must be safe to share
concurrently; query data cannot be retained. SpatialHash validates bounds/kinds and uses per-call scratch.

`SpatialHash()` is the native backend (`SpatialHash(cell_size)` selects a
finite positive width explicitly). Auto sizing uses the median positive maximum AABB
extent among edges/faces, excluding points. It uses a sparse `unordered_map` with full
signed int3 equality; each box covers every cell from `floor(lo/h)` through `floor(hi/h)`
inclusively. Records and bucket queries are parallel; map insertion is serial. The
lowest shared cell owns each pair, preventing duplicate output without skipping overlaps.

All input boxes are validated first. Nonfinite bounds, unrepresentable int64 cell
coordinates, or more than 2^20 box-cell records throw an overflow or length error. Invalid bounds/kinds still throw. There is **no pair-work
budget or pair-work-triggered fallback**. Long overlapping AABBs may be examined in
many shared cells despite unique output; dense buckets remain quadratic. Shared
backends are safe for concurrent calls because all query scratch is local.

The benchmark compares native SpatialHash contact with IPCTK.
The examples explicitly choose SpatialHash; contact sorts merged candidates deterministically. Swept bounds, inflation and
outward rounding are preserved. It has quadratic worst-case behavior for heavily overlapping boxes. Native local contact
evaluation, Hessian projection, gradient/Hessian assembly and surface CCD queries use TBB.
Assembly owns disjoint output DOFs and sparse columns, gathering contributions in fixed pair order;
scalar energy and CCD minimum reductions remain serial. Results are bitwise deterministic across thread counts.
A per-call vertex/pair incidence list uses O(vertices + active pairs) extra memory. Broadphase box
construction/sorting and active-set narrowphase remain serial. No acceleration-cache state is retained between evaluations.
The formulation follows [IPC (2020)](https://cims.nyu.edu/gcl/papers/2020-IPC.pdf) and
[Codimensional IPC / ACCD (2021)](https://arxiv.org/abs/2012.04457).

Migration: `BarrierContactEnergy` → `IPCTKContactEnergy`, `BarrierContactOptions` → `IPCTKContactOptions`,
`barrier_contact_energy.hpp` → `ipctk_contact_energy.hpp`, `litepgo::contact` → `litepgo::ipctk`, and
`LITEPGO_WITH_IPC` → `LITEPGO_WITH_IPC_TOOLKIT`. To use the native implementation instead, choose the
`IPCContactEnergy`/`IPCContactOptions` pair; the examples do this by default.

Migration: `SurfaceDofMap` → `SurfaceKinematics`, `surface_dof_map.hpp` → `surface_kinematics.hpp`
(C++ namespace and Python module remain `dof`; no old-name alias).
Surface kinematics accepts only `(rest_positions, W)`. For extracted boundaries,
`extract_surface_with_vertices` preserves exact volume vertex IDs; use
`dof::build_vertex_displacement_map(vertices, num_dofs, dofs_per_vertex = 3)`
(`litepgo/dof/vertex_displacement_map.hpp`) to build exact value-slot weights.
Negative/out-of-range IDs are rejected. Hermite uses width 24 and only value slots;
this does not sample curved faces. Independent surfaces use the locator path below.

`dof::SurfaceKinematics` (`litepgo/dof/surface_kinematics.hpp`) gives surface positions as an affine function of the
simulation DOFs, `p(u) = X + W u`, built from a general sparse `W`; zero rows keep static vertices at rest.
The barrier acts on `u` as `AffineMappedEnergy(barrier, map.displacement_map(), map.rest_positions())`, which
pulls gradients and Hessians back as `W^T g` and `W^T H W`. The contact examples preserve extracted vertex IDs to avoid inverse-solve roundoff
introducing tiny nonzero weights and expanding the mapped Hessian pattern.
`examples/contact/tet_p1_box_stack.cpp`
(and its Python mirror) presses two tet boxes together:

```cpp
const auto extracted = mesh::extract_surface_with_vertices(stack);
const auto& surface = extracted.mesh;
const auto W = dof::build_vertex_displacement_map(extracted.vertices, num_dofs);
auto barrier = std::make_shared<contact::IPCContactEnergy>(surface, contact::IPCContactOptions{.dhat = dhat},
    std::make_shared<contact::SpatialHash>());
const dof::SurfaceKinematics kinematics(surface.positions, W);
auto contact = std::make_shared<energy::AffineMappedEnergy>(barrier, kinematics.displacement_map(),
                                                            kinematics.rest_positions());
energy::WeightedEnergySum potential({{elastic, 1}, {loads, 1}, {contact, kappa}});
```

Python mirrors both backends as `litepgo.contact`; `IPCTKContactEnergy` and its options are exported only in toolkit-enabled builds. Friction and adaptive barrier stiffness are not implemented.
