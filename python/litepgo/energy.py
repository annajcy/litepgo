"""Energies and their composites, mirroring cpp/litepgo/energy.

Hessians are returned as scipy.sparse.csc_matrix. Before assembling a Hessian of
an energy whose pattern is not fixed, call update_pattern(x) at that x. Values are
always finite; evaluating at an infeasible state raises ValueError.

Numerical entry points are eval_value, eval_gradient, eval_hessian, and
eval_projected_hessian. eval(x, gradient=False, hessian=None) returns the value
and requested derivatives together.
"""

from litepgo._core import energy as _native

HessianKind = _native.HessianKind
Energy = _native.Energy
MeshEnergy = _native.MeshEnergy
QuadraticEnergy = _native.QuadraticEnergy
VertexPullingEnergy = _native.VertexPullingEnergy
WeightedEnergySum = _native.WeightedEnergySum
AffineMappedEnergy = _native.AffineMappedEnergy

__all__ = ["HessianKind", "Energy", "MeshEnergy", "QuadraticEnergy", "VertexPullingEnergy", "WeightedEnergySum", "AffineMappedEnergy"]


class BatchedMeshEnergy(_native.BatchedMeshEnergy):
    """Size-grouped vertex stencils; callbacks receive owned (batch, vertices, slots) arrays.

    Return (values, gradients, Hessians), with derivative shapes (B,m) and
    (B,m,m), or None when unrequested. Slots use vertex-major order.
    """
    def __init__(self, mapping, eval_batch, batch_size=256, step=None):
        import numpy as np
        slots = mapping.dofs_per_vertex

        def evaluate(ids, x, want_g, want_h):
            count, m = x.shape
            values, g, h = eval_batch(np.asarray(ids), x.reshape(count, m // slots, slots), want_g, want_h)
            if h is not None:
                h = np.asarray(h)
                if h.shape != (count, m, m):
                    raise ValueError("Expected batch Hessians with shape (B,m,m)")
                h = h.reshape(count * m, m)
            return values, g, h

        def bound(ids, x, dx):
            count, m = x.shape
            return step(np.asarray(ids), x.reshape(count, m // slots, slots), dx.reshape(count, m // slots, slots))

        super().__init__(mapping, evaluate, batch_size, bound if step is not None else None)


__all__ += ["BatchedMeshEnergy"]
