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
