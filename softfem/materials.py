"""Hyperelastic material models.

Every function is vectorised over a batch of deformation gradients ``F`` with
shape ``(m, 3, 3)``. Second derivatives use the row-major flattening
``vec(F)[3 * i + j] = F[i, j]`` and are returned with shape ``(m, 9, 9)``.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "lame",
    "cofactor",
    "cofactor_hessian",
    "project_psd",
    "LinearElastic",
    "StableNeoHookean",
    "cauchy_stress",
    "von_mises",
]

_I3 = np.eye(3)
_I9 = np.eye(9)
_LEVI = np.zeros((3, 3, 3))
_LEVI[0, 1, 2] = _LEVI[1, 2, 0] = _LEVI[2, 0, 1] = 1.0
_LEVI[0, 2, 1] = _LEVI[2, 1, 0] = _LEVI[1, 0, 2] = -1.0


def lame(E, nu):
    """Lamé parameters ``(mu, lambda)`` from Young's modulus and Poisson ratio.

    Scalars or per-element arrays are accepted.
    """
    E = np.asarray(E, dtype=float)
    nu = np.asarray(nu, dtype=float)
    if np.any(nu <= -1.0) or np.any(nu >= 0.5):
        raise ValueError("Poisson ratio must lie in (-1, 0.5)")
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    if mu.ndim == 0:
        return float(mu), float(lam)
    return mu, lam


def _per_element(a, extra_dims: int):
    """Reshape a scalar or ``(m,)`` parameter so it broadcasts against ``(m, ...)``."""
    a = np.asarray(a, dtype=float)
    return a.reshape(a.shape + (1,) * extra_dims) if a.ndim else a


def cofactor(F: np.ndarray) -> np.ndarray:
    """``dJ/dF`` for ``J = det(F)``; column ``j`` is ``f_{j+1} x f_{j+2}``."""
    f0, f1, f2 = F[..., :, 0], F[..., :, 1], F[..., :, 2]
    return np.stack([np.cross(f1, f2), np.cross(f2, f0), np.cross(f0, f1)], axis=-1)


def cofactor_hessian(F: np.ndarray) -> np.ndarray:
    """``d^2 J / dF^2`` with entries ``eps_ikm * eps_jln * F_mn``."""
    H = np.einsum("ikm,jln,emn->eijkl", _LEVI, _LEVI, F, optimize=True)
    return H.reshape(len(F), 9, 9)


def project_psd(H: np.ndarray, floor: float = 0.0) -> np.ndarray:
    """Clamp the eigenvalues of a batch of symmetric matrices to ``>= floor``."""
    w, V = np.linalg.eigh(H)
    np.maximum(w, floor, out=w)
    return (V * w[..., None, :]) @ np.swapaxes(V, -1, -2)


class LinearElastic:
    """Small-strain linear elasticity, ``Psi = mu e:e + lambda/2 tr(e)^2``.

    Not rotation invariant: use it for validation against linear theory only.
    """

    def __init__(self, E: float, nu: float):
        self.E, self.nu = float(E), float(nu)
        self.mu, self.lam = lame(E, nu)
        sym = np.einsum("ik,jl->ijkl", _I3, _I3) + np.einsum("il,jk->ijkl", _I3, _I3)
        vol = np.einsum("ij,kl->ijkl", _I3, _I3)
        self._H = (self.mu * sym + self.lam * vol).reshape(9, 9)

    def _strain(self, F):
        return 0.5 * (F + np.swapaxes(F, 1, 2)) - _I3

    def energy_density(self, F):
        e = self._strain(F)
        tr = np.trace(e, axis1=1, axis2=2)
        return self.mu * np.einsum("eij,eij->e", e, e) + 0.5 * self.lam * tr**2

    def pk1(self, F):
        e = self._strain(F)
        tr = np.trace(e, axis1=1, axis2=2)
        return 2.0 * self.mu * e + self.lam * tr[:, None, None] * _I3

    def hessian(self, F, project: bool = True):
        return np.broadcast_to(self._H, (len(F), 9, 9))


class StableNeoHookean:
    """Stable Neo-Hookean flesh model (Smith, de Goes & Kim 2018).

    ``Psi = mu/2 (I_C - 3) - mu (J - 1) + lambda'/2 (J - 1)^2`` in the form of
    Kim & Eberle's *Dynamic Deformables* notes. It is rest-stable, has no
    ``log J`` term (so inverted elements are handled gracefully) and uses
    ``lambda' = lambda + mu`` so that its linearisation equals linear elasticity.

    ``E`` and ``nu`` may be scalars or per-element arrays (heterogeneous bodies).
    """

    def __init__(self, E, nu):
        self.E = np.asarray(E, dtype=float) if np.ndim(E) else float(E)
        self.nu = np.asarray(nu, dtype=float) if np.ndim(nu) else float(nu)
        mu, lam = lame(E, nu)
        self.mu = mu
        self.lam = lam + mu

    def energy_density(self, F):
        J = np.linalg.det(F)
        Ic = np.einsum("eij,eij->e", F, F)
        return 0.5 * self.mu * (Ic - 3.0) - self.mu * (J - 1.0) + 0.5 * self.lam * (J - 1.0) ** 2

    def pk1(self, F):
        J = np.linalg.det(F)
        mu = _per_element(self.mu, 2)
        return mu * F + (self.lam * (J - 1.0) - self.mu)[:, None, None] * cofactor(F)

    def hessian(self, F, project: bool = True):
        J = np.linalg.det(F)
        c = cofactor(F).reshape(-1, 9)
        H = (
            _per_element(self.mu, 2) * _I9
            + _per_element(self.lam, 2) * c[:, :, None] * c[:, None, :]
            + (self.lam * (J - 1.0) - self.mu)[:, None, None] * cofactor_hessian(F)
        )
        return project_psd(H) if project else H


def cauchy_stress(F: np.ndarray, P: np.ndarray) -> np.ndarray:
    J = np.linalg.det(F)
    return (P @ np.swapaxes(F, 1, 2)) / J[:, None, None]


def von_mises(F: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Von Mises equivalent of the Cauchy stress for every element."""
    s = cauchy_stress(F, P)
    dev = s - np.trace(s, axis1=1, axis2=2)[:, None, None] / 3.0 * _I3
    return np.sqrt(1.5 * np.einsum("eij,eij->e", dev, dev))
