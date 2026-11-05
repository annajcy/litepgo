# Primitive groups and pair multipliers

`ContactPairWeights` supplies fixed integer group IDs for surface vertices,
edges and faces, plus a symmetric `G x G` multiplier table. Group IDs have no
built-in meaning: periodic ownership, component labels and other conventions
belong to the caller. There can be any number of groups.

## C++ API

```cpp
#include "litepgo/contact/contact_pair_weights.hpp"
#include "litepgo/contact/ipc_contact_energy.hpp"
#include "litepgo/contact/broad_phase/spatial_hash.hpp"

using namespace litepgo;
contact::ContactPairWeights weights;
weights.vertex_groups = vertex_tags;
weights.edge_groups = edge_tags;
weights.face_groups = face_tags;
weights.weights.resize(3, 3);
weights.weights << 1.0, 0.5, 0.0,
                   0.5, 1.0, 2.0,
                   0.0, 2.0, 1.0;
contact::IPCContactEnergy barrier(
    surface, options, std::make_shared<contact::SpatialHash>(), std::move(weights));
```

The fourth argument of native `IPCContactEnergy` is
`std::optional<ContactPairWeights>`, defaulting to `std::nullopt`.
For `IPCTKContactEnergy` it is the third argument, after the surface and Toolkit
options. Omission means multiplier one for every candidate. Existing calls keep
their behavior. Parameters are passed by value: lvalues are copied, rvalues can
be moved, and energies retain their own validated configuration without live
references to caller data. There is no runtime setter.

Omitting the argument or passing `std::nullopt` differs from explicitly passing
`ContactPairWeights{}`: that empty configuration has no valid table or matching
group arrays and is rejected during energy construction.

`contact_edges(surface.topology)` returns unique undirected edges as
`std::vector<std::array<int, 2>>`. Each row is `(min_vertex, max_vertex)` and rows
are sorted lexicographically. **`edge_groups` must use exactly this order.**
Vertex and face groups follow the input vertex and triangle order. Primitive
ownership cannot generally be inferred from endpoint tags: an owned face or
edge may contain ghost vertices in a periodic mesh.

## Python API

```python
from litepgo import contact

edges = contact.contact_edges(surface.topology)  # (E, 2) integer array
weights = contact.ContactPairWeights(
    vertex_groups=vertex_tags,
    edge_groups=edge_tags,
    face_groups=face_tags,
    weights=[[1.0, 0.5, 0.0], [0.5, 1.0, 2.0], [0.0, 2.0, 1.0]],
)
barrier = contact.IPCContactEnergy(
    surface, options, contact.SpatialHash(), pair_weights=weights,
)
```

`pair_weights=None` selects the original behavior. Python construction copies
the group sequences and real 2-D table; exposed configuration properties also
return copies and cannot be reassigned. Create a new configuration to change
groups or multipliers. The energy constructor checks all topology-dependent
conditions:

- Group-array lengths must equal `N`, `E`, and `F` for the input surface.
- Every group ID must be an integer in `[0, G)`.
- The table must be nonempty, square, exactly symmetric, finite and nonnegative.

Invalid configuration raises `std::invalid_argument` / Python `ValueError`;
Python also rejects noninteger group sequences at the binding boundary. A
positive candidate weight whose multiplication or accumulation is not representable
raises a domain error / `ValueError` during evaluation or snapshot generation.

## Where the multiplier acts

For an original VF candidate, look up the groups of its vertex and **original
face**. For EE, use its two **original edges**. A candidate contributes

$$
E_q = a_q\,W_{g_A,g_B}\,m_q\,b(s_q,h),
$$

where $a_q$ is the [reference-area weight](barrier.md), or one when area
weighting is disabled. $m_q$ is the EE mollifier, or one for VF. An external
stiffness can still multiply the complete energy through `WeightedEnergySum`.
All group multipliers are fixed with respect to positions, so value, gradient,
and exact/projected Hessians use the same weight.

Closest-feature reduction may turn different original candidates into the same
VV or VE record. Their **already scaled candidate weights are summed**. Applying
the table to reduced vertex IDs would lose the parent face/edge groups and give
the wrong answer. Native assembly does this directly. The optional Toolkit
backend groups original candidates by equal positive multiplier, builds each
subset with Toolkit, scales its contributions and merges identical records.
The number of Toolkit subsets depends on the distinct multipliers present.

## Zero, CCD and diagnostics

| Multiplier | Energy and derivatives | Active set, pattern and snapshot | CCD |
| --- | --- | --- | --- |
| Positive | Scale by that value | Retain eligible candidates, then merge | Enforce full minimum separation |
| Zero | No contribution | Exclude the original candidate | Skip that candidate |

A small positive multiplier does **not** relax clearance or enlarge the CCD
step. This differs from `mollifier=0`: a positive-weight mollified EE record
still retains all four support vertices even when its current mollifier is zero.

`snapshot.weights` includes the group multipliers, reference areas when enabled,
and merged multiplicities, while excluding stiffness and mollifiers. Counts,
minimum distance and pattern follow this same enabled active set; with all
contributions excluded, the count is zero and minimum distance is infinity.
A merged record has no unique original group pair.

The **initial whole-surface intersection check remains unconditional**, including
zero-multiplier pairs. Zero is not permission to start from an intersecting mesh.
CCD then bounds only enabled pairs, so a returned step does not certify that
excluded pairs stay separated throughout the motion. Callers using this for
periodic counting must ensure the band contains the required enabled copies.

See the [three-group example](../../examples/contact/pair-weights.md) for matching
C++/Python programs, CSV export, checks and a separate plot script.
