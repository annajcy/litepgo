# Fit network weights through equilibrium

[Back to PyTorch examples](./index.md)

This small CPU float64 example fits **two neural-network weights from equilibrium
displacements**. Torch supplies constitutive derivatives, native FEM assembles and
solves equilibrium, and the outer Torch optimizer receives an implicit loss gradient.
Newton iterations are not recorded in a Torch computation graph.

No training run or checkpoint is required. Keep the adjacent `neural_neohookean.py`
and `torch_neohookean.py` files with the example: it reuses their invariant network,
analytic density, and two-tetrahedron geometry. This example adds no library API.

## Run

From the repository root, after [configuring the release build](../../guide/build.md):

```sh
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/fitting/neural_equilibrium_fit.py
PYTHONPATH=build/release/python uv run --no-sync python -m pytest -q tests/python/test_neural_equilibrium_fit_example.py
```

The program prints JSON with initial/final loss and initial/fitted/reference weights.
It writes no files. One local run reduced the normalized loss from `0.5` to
approximately `1.7e-14`; final digits may differ across platforms. The test requires
at least a million-fold reduction, not exact recovery of the reference weights.

## Material and parameters

The density is

$$
W(F,F_p,p)=W_{\mathrm{NH}}(F,F_p;2,3)+W_{\mathrm{NN}}(F,F_p;\omega(p)).
$$

The fixed Neo-Hookean baseline gives this small untrained neural correction a
stable reference problem near rest. It is not a guarantee of global stability.
`W_NN` is the [existing invariant material](./neural-neohookean.md), including its
reference energy/stress corrections and plastic volume factor. Its architecture
is 3–32–32–1 with Softplus activations, initialized with seed 42. All weights except
the first two entries of the final linear layer remain fixed.

The three kinds of parameters have different roles:

| Quantity | Meaning | Changes during fitting? |
| --- | --- | --- |
| Physical `[lambda, mu]` | Fixed `[2, 3]` used by the baseline and original model interface | No |
| Captured module parameters | Network structure and fixed weight values | No |
| `p`, passed through material `theta` | Two actual final-layer weights, shared by all elements | Yes |

`num_parameters = 2` therefore counts **network weights**, not Lamé parameters.
The reference vector is packed by copying `model.net[4].weight[0, :2]`. `unpack(p)`
reconstructs the named parameter dictionary with a differentiable concatenation:

```python
return {
    **fixed,
    "net.4.weight": torch.cat((p, fixed["net.4.weight"][0, 2:]))[None],
}
```

The runtime does not call `functional_call` or replace module parameters. It
computes the fixed hidden layers with `torch.nn.functional.linear` and Softplus
on the whole batch. Let $h(z)$ be the 32 hidden features. At construction, it
computes constant $h_0=h(0)$ and $s=\partial h(0)/\partial z_3$. Each row then uses

$$
W_{\mathrm{NN},n}=\det(F_{p,n})\,w_n^T[h(z_n)-h_0-z_{n,3}s].
$$

The row's explicit active weights are concatenated with its fixed remaining
final-layer weights. This is algebraically the original reference-corrected
network, with no shared mutable model state, per-point Python loop, or lock.
Gradients flow through the explicit row weights and deformation inputs.
`unpack` remains a named-weight mapping for independent reference checks; it is
not used by the material hot path.

This implementation supports 1–32 active **final-layer** weights; the fitting
example uses two. It deliberately specializes the existing Linear/Softplus MLP.
The hidden layers must remain fixed: optimizing them would invalidate the cached
reference features/slopes and requires a different implementation.

## Independent rows, shared global weights

The material receives `theta` with shape `(N, K)`. Row `n` multiplies only its
own hidden features by its own weights, so batch execution preserves
$\partial W_n/\partial\theta_m=0$ for $n\ne m$. No row is broadcast from
`theta[0]`.

`ConstantDofMap(topology, 2)` expresses global weight sharing in the fitting
example. Native assembly accumulates independent point contributions into the
global parameter derivative. The benchmark uses the same material factory with
2, 8, or 32 active weights.

## Equilibrium and loss

The mesh has two tetrahedra and 15 displacement DOFs. Only DOFs `[9, 10, 11]` are
free; the other four vertices stay at rest. A fixed force `[0.015, -0.01, 0.02]`
acts on the free vertex, and `Fp = I` in both elements. The total potential is

$$
\Pi(u,p)=E_{\mathrm{internal}}(u,p)-f^T u.
$$

`make_energy(p)` returns that total potential. `make_response(u)` returns the
internal energy gradient as a function of weights at fixed displacement. Its
parameter derivative equals that of the total gradient because the external load
is independent of `p`.

The reference weights generate synthetic target displacements. The initial
weights are offset by `[0.15, -0.12]`. With a **fixed** scale equal to the initial
RMS free-displacement error, the objective is

$$
L(p)=\frac{1}{2|f|}\left\|\frac{u_f^*(p)-u_{f,\mathrm{target}}}{s}\right\|^2.
$$

Here the subscript $f$ denotes free DOFs. This normalization makes the small
network-induced displacement change visible; the initial loss is `0.5` by
construction. The example rejects an effectively zero initial observation scale.

`litepgo.torch.response(equilibrium, p)` connects the loss to
`StaticEquilibriumResponse`. At equilibrium, let $g_f=\partial\Pi/\partial u_f$,
$K_{ff}=\partial g_f/\partial u_f$, and $B_f=\partial g_f/\partial p$. Backward uses

$$
K_{ff}^T\lambda=\partial L/\partial u_f,\qquad
\frac{dL}{dp}=-B_f^T\lambda.
$$

The material supplies $\partial^2W/\partial F\partial p$. FEM assembles $B_f$;
the equilibrium response uses the exact, unshifted Hessian for sensitivities.
L-BFGS optimizes the two-entry tensor with up to 20 iterations. A failed equilibrium
raises an error; this example does not implement an outer infeasible-trial retry.

## What the tests establish

- Explicit weight evaluation matches independently loading a copied network, and
  material evaluations leave captured parameters unchanged.
- Material mixed derivatives and assembled fixed-state force derivatives match
  centered finite differences.
- The final loss gradient matches finite differences that **re-solve equilibrium**
  at each perturbed weight, at step sizes `1e-3` and `3e-4` (relative tolerance
  `2e-4`, absolute tolerance `2e-6`). Both selected weights have nonzero loss sensitivity.
- Reference and initial states satisfy the free equilibrium equations, with a
  positive definite exact free Hessian.
- Splitting the two elements into separate batches preserves the state, loss, and
  loss gradient; pointwise material comparisons also use distinct parameter rows.
- Optimization reduces the displacement loss. Weight uniqueness is not asserted.

## Limits

This is a local, noise-free demonstration with a stabilizing analytic baseline,
not full-network identification or evidence that an arbitrary learned density is
stable. There are no guarantees outside the tested load and parameter neighborhood.
Only the explicit two-entry tensor is optimized; the captured model remains fixed.
The outer bridge supports first derivatives, not double backward or GPU execution.

The current material path forms `9 x K` mixed derivatives at each volumetric
material point. With all network weights active, parameter replication and explicit
mixed derivatives become expensive. The equilibrium response also declares a dense
free-state-by-parameter sparsity pattern. Direct derivative products and larger
network training are separate scaling work, not capabilities demonstrated here.

::: details Complete Python source — `neural_equilibrium_fit.py`

<<< ../../../examples/fem/fitting/neural_equilibrium_fit.py

:::
