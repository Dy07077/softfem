"""Finite-element discretisation on linear tetrahedra and sparse assembly."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .materials import von_mises
from .mesh import TetMesh

__all__ = ["Elasticity", "HessianAssembler", "lumped_mass", "nodal_areas", "element_to_vertex"]


class Elasticity:
    """Elastic energy ``sum_e V_e Psi(F_e)`` with per-element constant ``F``.

    ``F = D_s D_m^{-1}`` is linear in the element's vertex positions,
    ``F_ij = sum_a x_ai G_aj``; ``B`` is the matching ``9 x 12`` matrix with
    ``vec(F) = B x_e``.
    """

    def __init__(self, mesh: TetMesh, material):
        self.mesh = mesh
        self.material = material
        X, T = mesh.X, mesh.T
        Dm = np.swapaxes(X[T[:, 1:]] - X[T[:, :1]], 1, 2)  # columns are edge vectors
        self.volume = np.linalg.det(Dm) / 6.0
        if np.any(self.volume <= 0.0):
            raise ValueError("mesh contains degenerate tetrahedra")
        Dm_inv = np.linalg.inv(Dm)
        G = np.empty((len(T), 4, 3))
        G[:, 1:, :] = Dm_inv
        G[:, 0, :] = -Dm_inv.sum(axis=1)
        self.G = G
        B = np.zeros((len(T), 3, 3, 4, 3))
        Gt = np.swapaxes(G, 1, 2)
        for i in range(3):
            B[:, i, :, :, i] = Gt
        self.B = B.reshape(len(T), 9, 12)
        self.Bt = np.ascontiguousarray(np.swapaxes(self.B, 1, 2))
        self.dofs = (3 * T[:, :, None] + np.arange(3)).reshape(len(T), 12)
        self.n_dof = 3 * mesh.n_nodes

    def deformation_gradient(self, x: np.ndarray) -> np.ndarray:
        return np.einsum("eai,eaj->eij", x[self.mesh.T], self.G, optimize=True)

    def energy(self, x: np.ndarray) -> float:
        return float(self.volume @ self.material.energy_density(self.deformation_gradient(x)))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Gradient w.r.t. the flattened positions, shape ``(3n,)``."""
        P = self.material.pk1(self.deformation_gradient(x))
        ge = np.einsum("eij,eaj->eai", P, self.G, optimize=True) * self.volume[:, None, None]
        return np.bincount(self.dofs.ravel(), ge.ravel(), minlength=self.n_dof)

    def hessian_blocks(self, x: np.ndarray, project: bool = True) -> np.ndarray:
        """Element stiffness matrices ``V B^T (d^2 Psi / dF^2) B``, ``(m, 12, 12)``."""
        H = self.material.hessian(self.deformation_gradient(x), project=project)
        return (self.Bt @ H @ self.B) * self.volume[:, None, None]

    def von_mises(self, x: np.ndarray) -> np.ndarray:
        F = self.deformation_gradient(x)
        return von_mises(F, self.material.pk1(F))


class HessianAssembler:
    """Scatter element ``12x12`` and nodal ``3x3`` blocks into one CSR pattern.

    The sparsity pattern is computed once; every assembly is two ``bincount``
    calls. Constrained degrees of freedom are eliminated by zeroing their rows
    and columns and putting ``1`` on the diagonal.
    """

    def __init__(self, n_nodes: int, tets: np.ndarray):
        n_dof = 3 * n_nodes
        dofs = (3 * tets[:, :, None] + np.arange(3)).reshape(-1, 12)
        er = np.broadcast_to(dofs[:, :, None], (len(dofs), 12, 12)).ravel()
        ec = np.broadcast_to(dofs[:, None, :], (len(dofs), 12, 12)).ravel()
        nd = 3 * np.arange(n_nodes)[:, None] + np.arange(3)
        nr = np.broadcast_to(nd[:, :, None], (n_nodes, 3, 3)).ravel()
        nc = np.broadcast_to(nd[:, None, :], (n_nodes, 3, 3)).ravel()
        keys = np.concatenate([er * n_dof + ec, nr * n_dof + nc])
        uniq, inv = np.unique(keys, return_inverse=True)
        self.n_dof = n_dof
        self.nnz = len(uniq)
        self.rows = uniq // n_dof
        self.cols = uniq % n_dof
        self.indices = self.cols.astype(np.int32)
        self.indptr = np.searchsorted(self.rows, np.arange(n_dof + 1)).astype(np.int32)
        self.elem_map = inv[: len(er)]
        self.node_map = inv[len(er) :]
        diag = np.arange(n_dof)
        self.diag_pos = np.searchsorted(uniq, diag * n_dof + diag)
        self._fixed = np.zeros(n_dof, dtype=bool)
        self._fixed_entries = np.zeros(self.nnz, dtype=bool)

    def set_fixed(self, fixed: np.ndarray) -> None:
        self._fixed = np.asarray(fixed, dtype=bool).copy()
        self._fixed_entries = self._fixed[self.rows] | self._fixed[self.cols]

    def assemble(self, elem_blocks=None, node_blocks=None) -> sp.csr_matrix:
        data = np.zeros(self.nnz)
        if elem_blocks is not None:
            data += np.bincount(self.elem_map, elem_blocks.ravel(), minlength=self.nnz)
        if node_blocks is not None:
            data += np.bincount(self.node_map, node_blocks.ravel(), minlength=self.nnz)
        if self._fixed.any():
            data[self._fixed_entries] = 0.0
            data[self.diag_pos[self._fixed]] = 1.0
        return sp.csr_matrix((data, self.indices, self.indptr), shape=(self.n_dof, self.n_dof))


def lumped_mass(mesh: TetMesh, density: float) -> np.ndarray:
    """Per-node mass: every tet gives a quarter of its mass to each vertex."""
    m = np.zeros(mesh.n_nodes)
    np.add.at(m, mesh.T.ravel(), np.repeat(density * mesh.volumes() / 4.0, 4))
    return m


def nodal_areas(X: np.ndarray, faces: np.ndarray, n_nodes: int) -> np.ndarray:
    """Surface area attributed to every node (a third of each incident triangle)."""
    a, b, c = X[faces[:, 0]], X[faces[:, 1]], X[faces[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    out = np.zeros(n_nodes)
    np.add.at(out, faces.ravel(), np.repeat(area / 3.0, 3))
    return out


def element_to_vertex(mesh: TetMesh, values: np.ndarray) -> np.ndarray:
    """Volume-weighted average of per-element values at the vertices."""
    w = mesh.volumes()
    num = np.zeros(mesh.n_nodes)
    den = np.zeros(mesh.n_nodes)
    np.add.at(num, mesh.T.ravel(), np.repeat(w * values, 4))
    np.add.at(den, mesh.T.ravel(), np.repeat(w, 4))
    return num / np.maximum(den, 1e-300)
