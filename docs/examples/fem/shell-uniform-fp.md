# Uniform plastic contraction: Fp = diag(1, 0.5, 1)

[Back to FEM basics](./index.md)

This simpler version of [free shell relaxation](./shell-fp-free-relaxation.md)
sets the **same** `Fp = diag(1, 0.5, 1)` on every face of the bundled Schwarz P
mesh. It uses native Koiter–StVK, thickness 0.06, E=1e8 and nu=0.35.
Start at the original rest shape and directly minimize shell energy, with no
continuation, periodic boundaries, contact or external load. All vertex coordinates are free, including rigid translations and rotations:
`DofReduction(X.size, [])` prescribes no fixed DOFs. The Hessian consequently
has rigid zero modes; the Newton solver may use diagonal shifts for its linear
solve, without adding a physical pinning energy.

```python
Fp = np.diag([1., .5, 1.])
shell = fem.ShellFemEnergy(
    fem.PointwiseShellAdapter[fem.KoiterStVK](), mapping, rest,
    [.06]*len(rest), [theta]*len(rest), Fp=[Fp]*len(rest),
)
```

![Uniform plastic contraction](/images/examples/shell-uniform-fp.png)

Unlike the spatially varying example, this Fp admits a compatible zero-energy
shape: `X @ Fp.T`, up to a rigid rotation and translation. The script compares
the relaxed mesh against that affine target after rigid alignment, and asserts
maximum vertex error below 1e-6 and full nodal residual below 1e-5. It does not
initialize the solve at the affine target.

On the tested Release build, the unconstrained solve converged in 30 iterations,
with energy 2.935e-18, full residual 1.325e-08, and maximum
aligned affine-target error 1.852e-12. OBJ exports and the
figure show the raw solver output, with no post-solve rigid alignment. Only the
validation target is aligned, because rigid pose is not uniquely determined.

From the litepgo repository root:

```sh
uv run --no-sync python examples/fem/material/shell_uniform_fp.py
uv run --no-sync python examples/fem/material/plot_shell_fp_free_relaxation.py --uniform
```

Outputs in `output/shell_uniform_fp/`: `original.obj`, `relaxed.obj`,
`affine_target.obj`, `relaxed.npz`, `summary.json`, and `comparison.png`.
No self-intersection check is performed.

<<< ../../../examples/fem/material/shell_uniform_fp.py
