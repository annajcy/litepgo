# Native IPC barrier formulation

The native contact energy evaluates barrier terms on packed surface positions.
This page defines the density and feature handling; [collision safety](collision-safety.md)
states surface validity and step constraints.

The energy retains its broad phase. A Python broad-phase subclass must not store
a strong reference back to the energy; use `weakref.ref(energy)` if needed. See
the [ownership contract](../concepts.md#python-ownership-must-remain-acyclic).

The Python `options` property returns a copy of the construction settings.
Construct a new contact energy to use different options.

## Barrier formulation

For candidate distance $d$, minimum separation $d_{\min}$, and activation
width $\hat d>0$, define

$$
s=d^2-d_{\min}^2,\qquad h=\hat d(\hat d+2d_{\min}),
$$

$$
b(s,h)=\begin{cases}-(s-h)^2\log(s/h),&0<s<h,\\0,&s\ge h.\end{cases}
$$

`dhat` and `min_distance` are physical lengths, with the same units as the
surface positions. `min_distance` corresponds to IPC Toolkit's `dmin`.
Activation starts at $d<d_{\min}+\hat d$, and feasibility requires
$d>d_{\min}$. Snapshot distances are unsquared $d$. The arguments $s,h$ of
the scalar barrier above are squared-length quantities.

### Raw and physical normalization

Both `IPCContactOptions` and `IPCTKContactOptions` expose
`use_physical_barrier`, defaulting to `false`, as in IPC Toolkit v1.6.1.
The unit-stiffness energy for either backend is

$$
C(x)=\sum_q A_q W_q m_q\,\gamma\,b(s_q,h),\qquad
\gamma=\begin{cases}
1,&\texttt{use\_physical\_barrier=false},\\
\hat d/h^2,&\texttt{use\_physical\_barrier=true}.
\end{cases}
$$

$A_q$ is the candidate reference-area weight or one, $W_q$ is its optional
group multiplier, and $m_q$ is the edge mollifier or one. The same constant
$\gamma$ multiplies the value, gradient and exact/projected Hessians.

Set stiffness $\kappa$ through `WeightedSumEnergy` to obtain $E=\kappa C$.
This has the same stiffness semantics as IPC Toolkit's `BarrierPotential`.
With reference-area weighting, physical mode gives $\kappa$ pressure units.
Raw mode gives energy/length$^6$ units. With $d_{\min}=0$,

$$
\gamma=1/\hat d^3,\qquad
\kappa_{\mathrm{physical}}=\kappa_{\mathrm{raw}}\hat d^3.
$$

For example, raw `dhat=0.001, kappa=1e16` and physical
`dhat=0.001, kappa=1e7` define the same energy with the same candidate weights.
Physical pressure units require area weighting. Unweighted mode retains a
candidate sum with different coefficient units.

```python
from litepgo import contact, energy

options = contact.IPCContactOptions(
    dhat=0.001, min_distance=0.0,
    use_area_weighting=True, use_physical_barrier=True,
)
barrier = contact.IPCContactEnergy(surface, options, contact.SpatialHash())
potential = energy.WeightedSumEnergy([(barrier, 1e7)])
```

In C++, set `options.use_physical_barrier = true` before constructing the
energy. The optional Toolkit backend accepts the same fields through
`IPCTKContactOptions`. Physical normalization follows the
[Toolkit potential implementation](https://github.com/ipc-sim/ipc-toolkit/blob/v1.6.1/src/ipc/potentials/barrier_potential.cpp)
and its
[stiffness units](https://github.com/ipc-sim/ipc-toolkit/blob/v1.6.1/src/ipc/potentials/barrier_potential.hpp).

The native energy sums nonincident vertex–triangle and unordered edge–edge
candidates. Closest-feature reduction uses point–point, point–edge,
point–triangle, or edge–edge distance as appropriate. Repeated feature stencils
are merged with their multiplicities preserved. Active-pair diagnostics count
merged stencils, not the original candidate multiplicity.

Nearly parallel edges use a rest-length mollifier. If $e_0,e_1$ are current
edge vectors and $\ell_{0,0},\ell_{1,0}$ their rest lengths,

$$
z=\frac{\|e_0\times e_1\|^2}{10^{-3}\ell_{0,0}^2\ell_{1,0}^2},\qquad
m(z)=\begin{cases}z(2-z),&z<1,\\1,&z\ge1.\end{cases}
$$

The contribution is $m(z)b(s,h)$. Derivatives include distance, mollifier,
and mixed product terms. Projected Hessians clamp each local term's eigenvalues.
Both `IPCContactOptions` and `IPCTKContactOptions` default to fixed reference-area
weighting. Set `use_area_weighting=False` to recover the original unweighted
candidate sum. With area weighting enabled:

- Each vertex and edge carries one third of the area of every incident reference face.
- A vertex-face candidate has weight `vertex_area / 4`.
- An edge-edge candidate has weight `(edge_area_0 + edge_area_1) / 4`.

The factors account for the two contacting surfaces and the VF/EE contributions.
They multiply the value, gradient and exact/projected Hessians consistently.
Closest-feature reduction and merging sum these weights, including repeated
lower-dimensional features. Areas stay fixed when the current positions change.
Area weighting does not change the activation distance, minimum separation or CCD.
It alone does not implement the full convergent IPC formulation or add friction.

Choose the normalization explicitly and set its coefficient directly through
`WeightedSumEnergy`. Calibrate it together with area weighting and the length
unit. `use_physical_barrier` affects energy scaling and derivatives. Activation,
closest-feature selection, reference-area weights and CCD use the same distances
in both modes. Snapshot weights exclude stiffness, normalization and mollifiers.

The [area-weighting example](../../examples/contact/area-weighting.md) compares
energy and normal force on fixed parallel sheets as the grid is refined.

Optional [primitive-group multipliers](pair-weights.md) multiply each original
candidate's area weight **before** closest-feature merging. Zero multipliers
exclude candidates from the active set and CCD; positive multipliers scale the
energy and its derivatives without relaxing minimum separation.
