# Coupling contact to displacement

Native contact consumes surface positions. A volume or shell solve usually
optimizes displacement, so connect the two with an affine map. Check
[surface validity and collision-safe steps](collision-safety.md) before solving.

## Connect contact to displacement DOFs

The constructor is `IPCContactEnergy(surface, options, broad_phase, planes=[])`.
The broad phase is required; `None`/null is rejected. Both languages retain shared
ownership of the backend. A typical volume boundary setup is:

```python
from litepgo import contact, dof, energy, mesh

extracted = mesh.extract_surface_with_vertices(volume)
surface = extracted.mesh
W = dof.build_vertex_displacement_map(extracted.vertices, num_dofs)
kinematics = dof.SurfaceKinematics(surface.positions, W)
barrier = contact.IPCContactEnergy(
    surface, contact.IPCContactOptions(dhat=0.01), contact.SpatialHash())
mapped_contact = energy.AffineMappedEnergy(
    barrier, kinematics.displacement_map, kinematics.rest_positions.ravel())
potential = energy.WeightedEnergySum([
    (elastic, 1.0), (pulling, 1.0), (mapped_contact, 1e3)])
```

Here `volume` is the rest volume mesh, `num_dofs` the displacement-vector length,
and `elastic`/`pulling` energies on those same DOFs. The
[two-box example](../../examples/contact.md) gives the complete construction and
boundary conditions. For Hermite vertex maps, pass `dofs_per_vertex=24`; to sample
between corners, use the [embedding path](../fem/embedding.md).

If $C(p)$ is the unit-stiffness barrier and $p=X_s+Wu$, then

$$
\Pi(u)=E_{\rm elastic}(u)+E_{\rm load}(u)+\kappa C(X_s+Wu),
$$

$$
g_c=\kappa W^T\nabla_p C,\qquad H_c=\kappa W^T\nabla_p^2C\,W.
$$

Set stiffness $\kappa$ through the energy weight. Contact Hessian sparsity
changes with active candidates; standalone calls must update the pattern.
Newton handles updates for the composed potential.
