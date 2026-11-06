# Triangle P1 sampling with a nonlinear Response

[Back to FEM basics](./index.md)

At a fixed reference point $\xi=(r,s)=(0.2,0.3)$, Triangle P1 gives
$N=(1-r-s,r,s)^T=(0.5,0.2,0.3)^T$. Interpolate three scalar coefficients and
exponentiate the result:

$$
z=N^Tc,\qquad \theta(c)=\exp(z),\qquad
J_\theta=\theta N^T,\qquad H_\theta(c,w)=w\theta NN^T.
$$

`AffineMappedResponse(N.transpose())` in C++, or `AffineMappedResponse(N[None, :])`
in Python, stores the sampling matrix for this query point and takes the
coefficients as input. C++ converts the matrix with `.sparseView()`. Its Hessian
pattern is an empty $3\times3$ matrix. `ComposedResponse(ExpResponse(1), sample)` supplies
the nonlinear derivative through the shared composition implementation.
The reference point stays fixed; no coordinate derivatives are requested.

For the outer energy $W(\theta)=\theta^2/2$, the example explicitly computes

$$
H_E=J_\theta^TJ_\theta+H_\theta(c,\theta)
   =2\theta^2NN^T.
$$

Both terms contribute equally. This demonstrates the
[weighted Hessian API](../../guide/energy/response-hessians.md) without changing
Field or introducing an Energy adapter. It evaluates one point, not a quadrature
integral, so it applies no geometric measure.

## Run

```sh
cmake --build build/release --target response_hessian litepgo_python
build/release/examples/fem/response/response_hessian
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/response/response_hessian.py
```

For $c=(-0.2,0.1,0.4)^T$, both programs check the analytic matrices and print:

```text
theta=1.040811, response H00=0.260203, energy H00=0.541644
```

::: details Complete Python source — `response_hessian.py`

<<< ../../../examples/fem/response/response_hessian.py

:::

::: details Complete C++ source — `response_hessian.cpp`

<<< ../../../examples/fem/response/response_hessian.cpp

:::
