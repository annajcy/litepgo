# Fit a shell material written in Torch

[Back to PyTorch examples](./index.md)

The Koiter shell density is written in PyTorch as a `TorchShellMaterial` and fills
the Material slot of `fem.ShellFemEnergy` and `fem.ShellFemForceMaterialResponse`.
The flat patch, prescribed boundary motion, reaction measurements and
Gauss–Newton fit are imported unchanged from the native
[shell equilibrium material fit](../fitting/shell-equilibrium-material-fit.md);
its `make_factories(problem, material)` takes the material class as an argument.
Keep both files together.

With natural forms $\bar a,\bar b$, thickness $t$ and
$\theta=(\lambda_{ps},\mu)$,

$$
\psi(S)=\frac{\lambda_{ps}}2[\operatorname{tr}(\bar a^{-1}S)]^2
+\mu\operatorname{tr}(\bar a^{-1}S\bar a^{-1}S),\qquad
W=t\,\psi\!\left(\tfrac12(a-\bar a)\right)+\frac{t^3}{12}\psi(b-\bar b).
$$

`W` receives packed forms $(a_{11},a_{12},a_{22},b_{11},b_{12},b_{22})$ for a
batch of faces and builds symmetric 2×2 matrices from them, reusing the single
off-diagonal entry. Autograd through the packed vectors then yields the
independent, unscaled derivatives the shell assembly expects. Each equilibrium
solve asks for $W$, $\partial W/\partial s$ and $\partial^2W/\partial s^2$; each
reaction Jacobian also asks for $\partial^2W/\partial s\,\partial\theta$.

The script fits $\theta$ from the initial guess $(0.7,1.2)$ to the generating
$(2,3)$ with the Torch material, repeats the fit with native `KoiterStVK`, and
prints the parameter error of the Torch fit and the difference between the two
fits. See [materials written in Python](../../guide/fem/python-materials.md#shell-materials).

## Run and inspect

```sh
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/fitting/torch_shell_material_fit.py
```

::: details Complete Python source — `torch_shell_material_fit.py`

<<< ../../../examples/fem/fitting/torch_shell_material_fit.py

:::
