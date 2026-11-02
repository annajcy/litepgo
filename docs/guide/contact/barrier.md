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
The model is the original unweighted barrier, not an area-weighted physical
barrier or a friction model.
