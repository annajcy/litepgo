# Value, gradient and Hessian coefficient sampling

The example fixes two Hex Q1 reference points and uses coefficients representing

$$
f(\xi)=\xi_1\xi_2,\qquad g(\xi)=\xi_2\xi_3.
$$

It constructs all three sampling factories and checks values, reference gradients,
and packed reference Hessians. The only nonzero independent second derivatives
are $f_{12}=1$ and $g_{23}=1$.

It also checks JVP/VJP against the returned sampling matrix and verifies that
the sampling Response's coefficient Hessian is zero, even though its sampled
spatial Hessians are nonzero. All native checks run in Release builds.

See [formulation and packing](../../guide/fem/sampling-responses.md) for the
point/derivative/component ordering and the distinction from spatial interpolation.

## Run

```sh
cmake --build build/release --target sampling_response litepgo_python
build/release/examples/fem/basis/sampling_response
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/basis/sampling_response.py
```

Both programs print:

```text
coefficient value, gradient and Hessian sampling: verified
```

::: details Python source
<<< ../../../examples/fem/basis/sampling_response.py
:::

::: details C++ source
<<< ../../../examples/fem/basis/sampling_response.cpp
:::

The examples also sample deformation gradients. Geometry scales reference axes
by `(2,3,4)`, while displacement coefficients interpolate `u(xi)=0.1*xi`.
Thus every sample is `diag(1.05, 1+0.1/3, 1.025)`. The factory takes displacement
coefficients, includes the identity offset, and returns column-major `vec(F)`
per point. It applies neither physical DOF transforms nor integration weights.

## From samples to an integrated energy

The final section fixes Gauss points, constructs the deformation-gradient sampling
Response, and evaluates Neo-Hookean material derivatives at its samples. It uses
physical weights from `VolumetricElement3D` and checks value, gradient and exact
Hessian against `VolumetricFemEnergy` on the same Hex Q1 element:

$$
E(c)=\sum_q w_q W(F_q),\qquad
 g_c=\sum_q w_q B_q^T\operatorname{vec}(P_q),\qquad
 H_c=\sum_q w_q B_q^T A_qB_q.
$$

Q1 nodal displacements equal the reference coefficients in this example.
For Hermite, use `BasisDofTransform` before sampling and pull derivatives back
before global assembly. The weights include the geometry determinant exactly
once. The Hessian comparison uses `Exact`; projected solver approximations have
separate semantics. Neither spatial interpolation Response is needed during
this active-coefficient evaluation.
