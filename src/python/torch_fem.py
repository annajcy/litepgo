"""PyTorch constitutive energies and force responses with native FEM assembly."""

import numpy as np
import torch

from litepgo import dof, energy, fem

__all__ = [
    "TorchVolumetricFemEnergy",
    "TorchVolumetricFemMaterialResponse",
    "TorchVolumetricFemFpResponse",
]


class _Infeasible(ValueError):
    pass


def _finite(value):
    if not np.isfinite(value).all():
        raise _Infeasible("non-finite material inputs or derivatives")
    return value


def _positive_det(matrices, name):
    determinants = np.linalg.det(matrices)
    if (
        not np.isfinite(matrices).all()
        or not np.isfinite(determinants).all()
        or (determinants <= 0).any()
    ):
        raise _Infeasible(f"{name} requires a finite positive determinant")


def _sample_fields(fields, count, width=None):
    if len(fields) != count:
        raise ValueError("expected one geometry or material field per element")
    for field in fields:
        if not isinstance(field, energy.Response):
            raise TypeError("expected spatial Response entries")
        if field.num_outputs <= 0 or (
            width is not None and field.num_outputs != width
        ):
            raise ValueError("incompatible field component count")
        width = field.num_outputs
    return _finite(
        np.array(
            [field.eval_value(np.full(3, 0.25)) for field in fields], dtype=np.float64
        ).reshape(count, width or 0)
    )


class _TetGeometry:
    def __init__(self, mapping, fields):
        if not isinstance(mapping, dof.VertexDofMap):
            raise TypeError("expected a VertexDofMap")
        if mapping.dofs_per_vertex != 3 or any(
            len(mapping.dof_indices(e)) != 12 for e in range(mapping.num_elements)
        ):
            raise ValueError("Torch volumetric FEM requires TetP1 displacement DOFs")
        count = mapping.num_elements
        if len(fields) != count:
            raise ValueError("expected one geometry field per element")
        xi = np.full(3, 0.25)
        derivatives, weights = [], []
        for X in fields:
            if not isinstance(X, energy.Response):
                raise TypeError("expected spatial Response entries")
            if X.num_outputs != 3:
                raise ValueError("reference geometry requires three components")
            J = X.eval_jacobian(xi).toarray()
            _positive_det(J, "reference geometry")
            B = np.kron((fem.TetP1Basis.eval_dN_dxi(xi) @ np.linalg.inv(J)).T, np.eye(3))
            # Native vec(F) is column-major; Torch reshapes matrices row-major.
            derivatives.append(B.reshape(3, 3, 12).transpose(1, 0, 2).reshape(9, 12))
            weights.append(np.linalg.det(J) / 6)
        self.B = _finite(np.array(derivatives, dtype=np.float64).reshape(count, 9, 12))
        self.weights = np.array(weights, dtype=np.float64)

    def deformation(self, ids, x):
        F = np.eye(3) + np.einsum("bij,bj->bi", self.B[ids], x).reshape(-1, 3, 3)
        _positive_det(F, "deformation")
        return F


def _density(material, F, Fp, theta):
    W = material(F, Fp, theta)
    if not isinstance(W, torch.Tensor):
        raise TypeError("material must return a Tensor")
    if W.shape != (len(F),) or W.dtype != torch.float64 or W.device.type != "cpu":
        raise ValueError("material must return a CPU float64 tensor of shape (B,)")
    return W


class TorchVolumetricFemEnergy(energy.MeshEnergy):
    """TetP1/centroid FEM energy driven by ``material(F, Fp, theta) -> W``.

    All three field lists contain one spatial Response per element. Geometry has three
    components, Fp has nine (column-major), and theta has a common positive width.
    Their centroid samples are snapshotted at construction; rebuild to change them.
    ``displacement_map`` must be a VertexDofMap with four vertices per element and
    three DOFs per vertex. Inputs are displacements, not deformed positions.

    The material receives CPU float64 tensors shaped (B,3,3), (B,3,3), and (B,k)
    and returns (B,) energy densities per original reference volume. It owns the
    full constitutive formula, including any multiplicative split and det(Fp).
    Rows must be independent, deterministic, twice differentiable in F, and safe
    for concurrent calls. Keep material parameters/buffers fixed while using this
    energy; no module mode or dtype is changed automatically.

    Exact local Hessians are pulled back from F; Projected uses MeshEnergy's
    per-element eigenvalue clamp before assembly. The domain requires det(F)>0
    and det(Fp)>0. The feasible-step hook protects det(F); materials with extra
    state-dependent domain restrictions must override eval_feasible_step_batch.
    Only displacement derivatives are exposed, not network-parameter gradients.
    """

    def __init__(
        self,
        displacement_map,
        rest_position_fields,
        Fp,
        theta,
        *,
        material,
        batch_size=256,
    ):
        self._geometry = _TetGeometry(displacement_map, rest_position_fields)
        count = displacement_map.num_elements
        self._Fp = _sample_fields(Fp, count, 9).reshape(count, 3, 3).transpose(0, 2, 1)
        _positive_det(self._Fp, "Fp")
        self._theta = _sample_fields(theta, count)
        if not callable(material):
            raise TypeError("material must be callable")
        self.material = material
        super().__init__(displacement_map, batch_size=batch_size)

    def eval_batch(self, ids, x, need_gradient, need_hessian):
        ids = np.asarray(ids)
        try:
            F = self._geometry.deformation(ids, x)
        except _Infeasible:
            return None  # Native Newton rejects this infeasible trial.
        # Native callbacks may run under no_grad/inference_mode or on TBB threads.
        with (
            torch.inference_mode(False),
            torch.set_grad_enabled(need_gradient or need_hessian),
        ):
            F = torch.tensor(
                F,
                dtype=torch.float64,
                device="cpu",
                requires_grad=need_gradient or need_hessian,
            )
            W = _density(
                self.material,
                F,
                torch.tensor(self._Fp[ids], device="cpu"),
                torch.tensor(self._theta[ids], device="cpu"),
            )
            values = W.detach().numpy().copy() * self._geometry.weights[ids]
            if not np.isfinite(values).all():
                return None
            gradient = hessian = None
            if need_gradient or need_hessian:
                # A zero quadratic also makes constant/linear densities differentiable twice.
                W = W + (0.0 * F).square().sum(dim=(1, 2))
                stress = torch.autograd.grad(W.sum(), F, create_graph=need_hessian)[0]
                B = self._geometry.B[ids]
                if need_gradient:
                    gradient = self._geometry.weights[ids, None] * np.einsum(
                        "bij,bi->bj", B, stress.detach().numpy().reshape(-1, 9)
                    )
                if need_hessian:
                    rows = [
                        torch.autograd.grad(
                            stress.reshape(-1, 9)[:, i].sum(), F, retain_graph=i < 8
                        )[0].reshape(-1, 9)
                        for i in range(9)
                    ]
                    tangent = torch.stack(rows, dim=1).detach().numpy()
                    hessian = (
                        self._geometry.weights[ids, None, None]
                        * np.einsum("bik,bij,bjl->bkl", B, tangent, B)
                    ).reshape(-1, 12)
        if any(a is not None and not np.isfinite(a).all() for a in (gradient, hessian)):
            return None
        return values, gradient, hessian

    def eval_feasible_step_batch(self, ids, x, dx):
        ids = np.asarray(ids)
        F = self._geometry.deformation(ids, x)
        dF = np.einsum("bij,bj->bi", self._geometry.B[ids], dx).reshape(-1, 3, 3)
        return np.array(
            [fem.eval_positive_det_feasible_step(a, da) for a, da in zip(F, dF)]
        )


class _TorchForceResponse(energy.MeshResponse):
    def __init__(
        self,
        parameter_map,
        displacement_map,
        rest_position_fields,
        displacement,
        fixed_fields,
        *,
        plastic_input,
        material,
        batch_size,
    ):
        self._geometry = _TetGeometry(displacement_map, rest_position_fields)
        count = displacement_map.num_elements
        if not isinstance(parameter_map, (dof.ConstantDofMap, dof.ElementDofMap)):
            raise TypeError("parameter_map must be ConstantDofMap or ElementDofMap")
        if parameter_map.num_elements != count:
            raise ValueError("parameter and displacement element counts must match")
        width = (
            parameter_map.num_dofs
            if isinstance(parameter_map, dof.ConstantDofMap)
            else parameter_map.dofs_per_element
        )
        if width <= 0 or (plastic_input and width != 9):
            raise ValueError("parameter width must be positive (nine for Fp)")
        displacement = np.asarray(displacement, dtype=np.float64)
        if displacement.shape != (displacement_map.num_dofs,):
            raise ValueError("incompatible displacement shape")
        _finite(displacement)
        local = np.array(
            [displacement_map.gather(e, displacement) for e in range(count)]
        ).reshape(count, 12)
        self._F = self._geometry.deformation(np.arange(count), local)
        self._fixed = _sample_fields(fixed_fields, count, None if plastic_input else 9)
        if not plastic_input:
            self._fixed = self._fixed.reshape(count, 3, 3).transpose(0, 2, 1)
            _positive_det(self._fixed, "Fp")
        if not callable(material):
            raise TypeError("material must be callable")
        self.material = material
        self._plastic_input = plastic_input
        super().__init__(parameter_map, displacement_map, batch_size=batch_size)

    def _inputs(self, ids, x, derivatives):
        _finite(x)
        if self._plastic_input:
            _positive_det(x.reshape(-1, 3, 3).transpose(0, 2, 1), "Fp")
        p = torch.tensor(
            x, dtype=torch.float64, device="cpu", requires_grad=derivatives
        )
        F = torch.tensor(self._F[ids], device="cpu", requires_grad=True)
        fixed = torch.tensor(self._fixed[ids], device="cpu")
        Fp, theta = (
            (p.reshape(-1, 3, 3).transpose(1, 2), fixed)
            if self._plastic_input
            else (fixed, p)
        )
        W = _density(self.material, F, Fp, theta)
        _finite(W.detach().numpy())
        # Keep zero mixed derivatives connected for constant/parameter-independent W.
        W = W + (0.0 * F).square().sum(dim=(1, 2)) * (1 + (0.0 * p).square().sum(dim=1))
        return W, F, p

    def _pullback(self, ids, stress):
        return _finite(
            self._geometry.weights[ids, None]
            * np.einsum(
                "bij,bi->bj",
                self._geometry.B[ids],
                stress.detach().numpy().reshape(-1, 9),
            )
        )

    def eval_batch(self, ids, x, need_jacobian):
        ids = np.asarray(ids)
        with torch.inference_mode(False), torch.enable_grad():
            try:
                W, F, p = self._inputs(ids, x, need_jacobian)
                stress = torch.autograd.grad(W.sum(), F, create_graph=need_jacobian)[0]
                value = self._pullback(ids, stress)
                if not need_jacobian:
                    return value
                rows = [
                    torch.autograd.grad(
                        stress.reshape(-1, 9)[:, i].sum(), p, retain_graph=i < 8
                    )[0]
                    for i in range(9)
                ]
                mixed = torch.stack(rows, dim=1).detach().numpy()
                J = self._geometry.weights[ids, None, None] * np.einsum(
                    "bij,bik->bjk", self._geometry.B[ids], mixed
                )
                return value, _finite(J.reshape(-1, x.shape[1]))
            except _Infeasible:
                return None

    def eval_jvp_batch(self, ids, x, dx):
        ids = np.asarray(ids)
        with torch.inference_mode(False), torch.enable_grad():
            W, F, p = self._inputs(ids, x, True)
            # d/dF (dW/dp · dx), without forming the mixed derivative matrix.
            stress, parameter_gradient = torch.autograd.grad(
                W.sum(), (F, p), create_graph=True
            )
            self._pullback(ids, stress)  # Validate the force at this state too.
            product = torch.autograd.grad(
                parameter_gradient, F, grad_outputs=torch.tensor(dx, device="cpu")
            )[0]
            return self._pullback(ids, product)

    def eval_vjp_batch(self, ids, x, gy):
        ids = np.asarray(ids)
        with torch.inference_mode(False), torch.enable_grad():
            W, F, p = self._inputs(ids, x, True)
            stress = torch.autograd.grad(W.sum(), F, create_graph=True)[0]
            self._pullback(ids, stress)
            weighted = self._geometry.weights[ids, None] * np.einsum(
                "bij,bj->bi", self._geometry.B[ids], gy
            )
            # d/dp (dW/dF · (weight B gy)); native scatter sums shared parameters.
            product = torch.autograd.grad(
                stress,
                p,
                grad_outputs=torch.tensor(weighted.reshape(-1, 3, 3), device="cpu"),
            )[0]
            return _finite(product.detach().numpy().copy())


class TorchVolumetricFemMaterialResponse(_TorchForceResponse):
    """Fixed-state positive-gradient force as a function of physical theta.

    Constructor conventions match TorchVolumetricFemEnergy. ConstantDofMap shares
    a theta block; ElementDofMap supplies one block per element. Active parameters
    are constant within each element. Geometry, displacement, and Fp centroid
    samples are snapshotted. Output is the assembled global displacement gradient.
    Jacobians are explicit; JVP/VJP directly contract mixed Torch derivatives.
    W must be twice differentiable jointly in F and the active parameters.
    Network weights are fixed captures, not implicit optimization inputs.
    """

    def __init__(
        self,
        parameter_map,
        displacement_map,
        rest_position_fields,
        displacement,
        Fp,
        *,
        material,
        batch_size=256,
    ):
        super().__init__(
            parameter_map,
            displacement_map,
            rest_position_fields,
            displacement,
            Fp,
            plastic_input=False,
            material=material,
            batch_size=batch_size,
        )


class TorchVolumetricFemFpResponse(_TorchForceResponse):
    """Fixed-state positive-gradient force as a function of column-major vec(Fp).

    ConstantDofMap shares nine inputs; ElementDofMap assigns nine per element.
    Fixed theta fields, geometry, and displacement are snapshotted at construction.
    Inputs must have positive det(Fp). Compose with LogDiagonalFpResponse (and
    ElementwiseResponse for element blocks) to optimize positive diagonal stretches.
    Shares the energy's reference-volume convention and native assembly. Explicit
    Jacobian and direct mixed-derivative JVP/VJP are provided. W must be twice
    differentiable jointly in F and Fp; captured network weights stay fixed.
    """

    def __init__(
        self,
        parameter_map,
        displacement_map,
        rest_position_fields,
        displacement,
        theta,
        *,
        material,
        batch_size=256,
    ):
        super().__init__(
            parameter_map,
            displacement_map,
            rest_position_fields,
            displacement,
            theta,
            plastic_input=True,
            material=material,
            batch_size=batch_size,
        )
