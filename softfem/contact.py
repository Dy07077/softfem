"""Frictional contact against analytic obstacles with an IPC-style barrier.

Non-penetration follows Incremental Potential Contact (Li et al., SIGGRAPH
2020): every surface node ``i`` adds ``kappa * A_i * b(d_i)`` to the
incremental potential, where ``d_i`` is its distance to an obstacle and

    b(d) = -(d - dhat)^2 ln(d / dhat)      for 0 < d < dhat, else 0.

The barrier is C2, convex in ``d`` and unbounded as ``d -> 0``; combined with a
line search that is filtered by continuous collision detection no node ever
crosses an obstacle. Contact is node-based: against a plane that keeps whole
faces out too (a half-space is convex), but edges and faces are not tested
against spheres. Friction uses IPC's lagged, smoothed Coulomb potential.
"""
from __future__ import annotations

import numpy as np

__all__ = ["barrier", "friction_f0", "friction_f1_over_y", "Plane", "Sphere", "ContactSet", "Friction"]


def barrier(d, dhat):
    """Return ``b(d)``, ``b'(d)`` and ``b''(d)`` (element-wise, unit stiffness)."""
    d = np.asarray(d, dtype=float)
    b = np.zeros_like(d)
    b1 = np.zeros_like(d)
    b2 = np.zeros_like(d)
    m = d < dhat
    if np.any(m):
        dm = d[m]
        r = dm - dhat
        L = np.log(dm / dhat)
        b[m] = -r * r * L
        b1[m] = -(2.0 * r * L + r * r / dm)
        b2[m] = -(2.0 * L + 4.0 * r / dm - (r / dm) ** 2)
    return b, b1, b2


def friction_f0(y, eps):
    """Smoothed ``|u|``: C1 at ``eps`` with ``f0(0) = eps / 3``."""
    y = np.asarray(y, dtype=float)
    s = y < eps
    out = y.copy()
    ys = y[s]
    out[s] = -ys**3 / (3.0 * eps**2) + ys**2 / eps + eps / 3.0
    return out


def friction_f1_over_y(y, eps):
    """``f0'(y) / y``, finite at ``y = 0``."""
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    s = y < eps
    out[s] = 2.0 / eps - y[s] / eps**2
    out[~s] = 1.0 / y[~s]
    return out


def _tangent_basis(n: np.ndarray) -> np.ndarray:
    """Orthonormal ``(k, 3, 2)`` tangent frames for unit normals ``(k, 3)``."""
    a = np.zeros_like(n)
    use_y = np.abs(n[:, 0]) > 0.9
    a[~use_y, 0] = 1.0
    a[use_y, 1] = 1.0
    t1 = a - np.einsum("ij,ij->i", a, n)[:, None] * n
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(n, t1)
    return np.stack([t1, t2], axis=2)


class Plane:
    """Infinite rigid plane; bodies stay on the side the normal points to."""

    def __init__(self, point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), mu: float = 0.0):
        n = np.asarray(normal, dtype=float)
        self.normal_vec = n / np.linalg.norm(n)
        self.point = np.asarray(point, dtype=float)
        self.mu = float(mu)

    def distance(self, x):
        return (x - self.point) @ self.normal_vec

    def normal(self, x):
        return np.broadcast_to(self.normal_vec, x.shape)

    def distance_hessian(self, x):
        return None

    def max_step(self, x, dx, shrink=0.9):
        d0 = self.distance(x)
        vn = dx @ self.normal_vec
        m = vn < 0.0
        if not np.any(m):
            return np.inf
        return float(np.min(shrink * d0[m] / -vn[m]))


class Sphere:
    """Rigid ball obstacle."""

    def __init__(self, center=(0.0, 0.0, 0.0), radius: float = 1.0, mu: float = 0.0):
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)
        self.mu = float(mu)

    def distance(self, x):
        return np.linalg.norm(x - self.center, axis=1) - self.radius

    def normal(self, x):
        r = x - self.center
        return r / np.linalg.norm(r, axis=1, keepdims=True)

    def distance_hessian(self, x):
        r = x - self.center
        ln = np.linalg.norm(r, axis=1)
        n = r / ln[:, None]
        return (np.eye(3) - n[:, :, None] * n[:, None, :]) / ln[:, None, None]

    def max_step(self, x, dx, shrink=0.9):
        rel = x - self.center
        d0 = np.linalg.norm(rel, axis=1) - self.radius
        target = self.radius + (1.0 - shrink) * d0
        a = np.einsum("ij,ij->i", dx, dx)
        b = 2.0 * np.einsum("ij,ij->i", rel, dx)
        c = np.einsum("ij,ij->i", rel, rel) - target**2
        disc = b * b - 4.0 * a * c
        hit = (disc >= 0.0) & (b < 0.0) & (a > 0.0)
        if not np.any(hit):
            return np.inf
        t = (-b[hit] - np.sqrt(disc[hit])) / (2.0 * a[hit])
        return float(np.min(t))


class ContactSet:
    """Barrier potential between a set of mesh nodes and rigid obstacles."""

    def __init__(self, obstacles, nodes, weights, dhat: float, kappa: float):
        self.obstacles = list(obstacles)
        self.nodes = np.asarray(nodes, dtype=np.int64)
        self.weights = np.asarray(weights, dtype=float)
        self.dhat = float(dhat)
        self.kappa = float(kappa)

    def distances(self, x):
        xs = x[self.nodes]
        return [o.distance(xs) for o in self.obstacles]

    def min_distance(self, x) -> float:
        ds = self.distances(x)
        return float(min((d.min() for d in ds), default=np.inf))

    def energy(self, x) -> float:
        e = 0.0
        for d in self.distances(x):
            if np.any(d <= 0.0):
                return np.inf
            b, _, _ = barrier(d, self.dhat)
            e += self.kappa * float(self.weights @ b)
        return e

    def gradient(self, x) -> np.ndarray:
        g = np.zeros_like(x)
        xs = x[self.nodes]
        for o in self.obstacles:
            _, b1, _ = barrier(o.distance(xs), self.dhat)
            g[self.nodes] += (self.kappa * self.weights * b1)[:, None] * o.normal(xs)
        return g

    def hessian_blocks(self, x, n_nodes: int) -> np.ndarray:
        H = np.zeros((n_nodes, 3, 3))
        xs = x[self.nodes]
        for o in self.obstacles:
            d = o.distance(xs)
            act = d < self.dhat
            if not np.any(act):
                continue
            _, b1, b2 = barrier(d[act], self.dhat)
            n = o.normal(xs[act])
            k = self.kappa * self.weights[act]
            blk = (k * b2)[:, None, None] * n[:, :, None] * n[:, None, :]
            dh = o.distance_hessian(xs[act])
            if dh is not None:
                blk = blk + (k * b1)[:, None, None] * dh
                w, V = np.linalg.eigh(blk)
                blk = (V * np.maximum(w, 0.0)[:, None, :]) @ np.swapaxes(V, 1, 2)
            np.add.at(H, self.nodes[act], blk)
        return H

    def max_step(self, x, p) -> float:
        xs, ps = x[self.nodes], p[self.nodes]
        return min((o.max_step(xs, ps) for o in self.obstacles), default=np.inf)

    def normal_forces(self, x):
        """Magnitude of the contact force on every node, one array per obstacle."""
        out = []
        for d in self.distances(x):
            _, b1, _ = barrier(d, self.dhat)
            out.append(-self.kappa * self.weights * b1)
        return out


class Friction:
    """Lagged smoothed Coulomb friction (IPC, Sec. 5).

    ``update`` freezes the normal force magnitudes ``lam`` and tangent frames
    ``T`` at the start of a time step; within the step the friction potential
    ``mu * lam * f0(|T^T (x - x_ref)|)`` is a smooth function of ``x``.
    """

    def __init__(self):
        self.groups = []
        self.eps = 1.0

    def update(self, contact: ContactSet, x_ref: np.ndarray, eps: float) -> None:
        self.groups = []
        self.eps = float(eps)
        xs = x_ref[contact.nodes]
        for o, lam_all in zip(contact.obstacles, contact.normal_forces(x_ref)):
            if o.mu <= 0.0:
                continue
            act = lam_all > 0.0
            if not np.any(act):
                continue
            idx = contact.nodes[act]
            T = _tangent_basis(np.ascontiguousarray(o.normal(xs[act])))
            self.groups.append((idx, o.mu * lam_all[act], T, x_ref[idx].copy()))

    def _tangential(self, x, group):
        idx, w, T, xr = group
        u = np.einsum("kij,ki->kj", T, x[idx] - xr)
        return u, np.linalg.norm(u, axis=1)

    def energy(self, x) -> float:
        e = 0.0
        for grp in self.groups:
            _, y = self._tangential(x, grp)
            e += float(grp[1] @ friction_f0(y, self.eps))
        return e

    def gradient(self, x) -> np.ndarray:
        g = np.zeros_like(x)
        for grp in self.groups:
            idx, w, T, _ = grp
            u, y = self._tangential(x, grp)
            gu = (w * friction_f1_over_y(y, self.eps))[:, None] * u
            np.add.at(g, idx, np.einsum("kij,kj->ki", T, gu))
        return g

    def hessian_blocks(self, x, n_nodes: int) -> np.ndarray:
        H = np.zeros((n_nodes, 3, 3))
        eps = self.eps
        for grp in self.groups:
            idx, w, T, _ = grp
            u, y = self._tangential(x, grp)
            M = np.zeros((len(idx), 2, 2))
            s = y < eps
            I2 = np.eye(2)
            if np.any(s):
                ys = y[s]
                us = u[s]
                inv = np.divide(1.0, ys, out=np.zeros_like(ys), where=ys > 0)
                M[s] = (2.0 / eps - ys / eps**2)[:, None, None] * I2 - (inv / eps**2)[:, None, None] * (
                    us[:, :, None] * us[:, None, :]
                )
            if np.any(~s):
                yl = y[~s]
                uh = u[~s] / yl[:, None]
                M[~s] = (I2 - uh[:, :, None] * uh[:, None, :]) / yl[:, None, None]
            blk = w[:, None, None] * (T @ M @ np.swapaxes(T, 1, 2))
            np.add.at(H, idx, blk)
        return H
