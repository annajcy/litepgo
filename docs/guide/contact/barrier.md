# Native IPC barrier formulation

The native contact energy evaluates barrier terms on packed surface positions.
This page defines the density and feature handling; [collision safety](collision-safety.md)
states surface validity and step constraints.

## Barrier formulation

For candidate distance $d$, minimum separation $d_{\min}$, and activation
width $\hat d>0$, define

$$
s=d^2-d_{\min}^2,\qquad h=\hat d(\hat d+2d_{\min}),
$$

$$
b(s,h)=\begin{cases}-(s-h)^2\log(s/h),&0<s<h,\\0,&s\ge h.\end{cases}
$$

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

The raw barrier formula above is unchanged by area weighting. Set its coefficient
directly through `WeightedSumEnergy`; no normalization conversion is applied.
Calibrate the coefficient for the barrier definition, area weighting and length
unit being used.

The [area-weighting example](../../examples/contact/area-weighting.md) compares
energy and normal force on fixed parallel sheets as the grid is refined.

Optional [primitive-group multipliers](pair-weights.md) multiply each original
candidate's area weight **before** closest-feature merging. Zero multipliers
exclude candidates from the active set and CCD; positive multipliers scale the
energy and its derivatives without relaxing minimum separation.
