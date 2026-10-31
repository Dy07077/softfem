"""Backward-Euler and quasi-static solves as energy minimisation.

Each step minimises the incremental potential

    E(x) = 1/(2 h^2) |x - x_tilde|_M^2 + Psi(x) + B(x) + D(x),
    x_tilde = x_n + h v_n + h^2 g,

(or ``Psi + B + D - f_ext . x`` in quasi-static mode) with projected Newton:
element Hessians are projected to be positive semi-definite, the step is
capped by continuous collision detection and then backtracked until the
Armijo condition holds. Constrained coordinates are eliminated from the
linear system, so prescribed motions are satisfied exactly. They bypass the
line search, so their straight path over each step goes through the same CCD
instead, and a step that would carry a node into an obstacle is refused.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse.linalg as spla

from .contact import ContactSet, Friction
from .fem import Elasticity, HessianAssembler, element_to_vertex, lumped_mass, nodal_areas
from .mesh import TetMesh

__all__ = ["Simulator", "StepInfo"]


@dataclass
class StepInfo:
    iterations: int = 0
    converged: bool = False
    residuals: list = field(default_factory=list)  # |p|_inf of each Newton direction
    alphas: list = field(default_factory=list)  # accepted line-search step sizes
    seconds: float = 0.0


class Simulator:
    """Implicit FEM simulation of soft bodies in contact with rigid obstacles.

    Args:
        mesh: tetrahedral mesh (several disconnected bodies are fine).
        material: e.g. :class:`softfem.StableNeoHookean`.
        density: mass density in kg/m^3.
        dt: time step. In quasi-static mode it only scales friction smoothing.
        gravity: acceleration vector.
        obstacles: rigid :class:`Plane` / :class:`Sphere` objects.
        dhat: barrier activation distance (default: 1e-3 of the bbox diagonal).
        kappa: barrier stiffness (default: ``0.1 E / dhat``).
        eps_v: friction velocity threshold below which friction is smoothed.
        quasistatic: drop inertia and solve for static equilibrium each step.
        newton_tol: stop when ``|p|_inf < newton_tol * bbox_diagonal``.
        regularization: diagonal shift added in quasi-static mode, only needed
            when constraints leave rigid motions free.
    """

    def __init__(
        self,
        mesh: TetMesh,
        material,
        *,
        density: float = 1000.0,
        dt: float = 1e-2,
        gravity=(0.0, 0.0, -9.81),
        obstacles=(),
        dhat: float | None = None,
        kappa: float | None = None,
        eps_v: float | None = None,
        quasistatic: bool = False,
        newton_tol: float = 1e-5,
        max_newton: int = 60,
        contact_nodes=None,
        contact_weights=None,
        project: bool = True,
        regularization: float = 0.0,
    ):
        self.mesh = mesh
        self.material = material
        self.elastic = Elasticity(mesh, material)
        self.n = mesh.n_nodes
        self.mass = lumped_mass(mesh, density)
        self.dt = float(dt)
        self.gravity = np.asarray(gravity, dtype=float)
        self.quasistatic = bool(quasistatic)
        self.project = project
        self.regularization = float(regularization)
        L = mesh.bbox_diagonal()
        self.length_scale = L
        self.dhat = float(dhat) if dhat is not None else 1e-3 * L
        self.kappa = float(kappa) if kappa is not None else 0.1 * float(np.mean(material.E)) / self.dhat
        self.eps_v = float(eps_v) if eps_v is not None else 1e-3 * L
        self.newton_tol = newton_tol * L
        self.max_newton = int(max_newton)
        if contact_nodes is None:
            faces = mesh.boundary_faces()
            contact_nodes = np.unique(faces)
            contact_weights = nodal_areas(mesh.X, faces, self.n)[contact_nodes]
        elif contact_weights is None:
            contact_weights = np.ones(len(contact_nodes))
        self.contact = ContactSet(obstacles, contact_nodes, contact_weights, self.dhat, self.kappa)
        self.friction = Friction()
        self.assembler = HessianAssembler(self.n, mesh.T)
        self.x = mesh.X.copy()
        self.v = np.zeros_like(self.x)
        self.t = 0.0
        self.fixed = np.zeros((self.n, 3), dtype=bool)
        self._constraints = []
        self._x_tilde = self.x.copy()
        self.history: list[StepInfo] = []

    # ------------------------------------------------------------------ setup
    def fix(self, nodes, target=None, axes=(0, 1, 2)) -> None:
        """Prescribe coordinates ``axes`` of ``nodes``.

        ``target`` is ``None`` (hold the current position), a ``(k, 3)`` array,
        or a callable ``t -> (k, 3)`` evaluated at the end of every step.
        Each step moves the nodes along a straight line to the new target; if
        that line enters an obstacle, :meth:`step` raises ``RuntimeError`` and
        leaves the state unchanged.
        """
        nodes = np.asarray(nodes, dtype=np.int64)
        if target is None:
            target = self.x[nodes].copy()
        self._constraints.append((nodes, tuple(axes), target))
        for a in axes:
            self.fixed[nodes, a] = True
        self.assembler.set_fixed(self.fixed.ravel())

    def _apply_constraints(self, x: np.ndarray, t: float) -> None:
        for nodes, axes, target in self._constraints:
            pos = np.asarray(target(t) if callable(target) else target, dtype=float)
            for a in axes:
                x[nodes, a] = pos[:, a]

    def _check_prescribed_path(self, x0: np.ndarray, x1: np.ndarray) -> None:
        """Refuse prescribed motion whose straight path from ``x0`` to ``x1`` enters an obstacle.

        Both ends can be outside an obstacle while the path between them is not
        (a node carried through a sphere), so checking the targets alone is not enough.
        """
        nodes = self.contact.nodes[self.fixed[self.contact.nodes].any(axis=1)]
        if len(nodes) == 0:
            return
        dx = x1[nodes] - x0[nodes]
        for k, o in enumerate(self.contact.obstacles):
            if o.max_step(x0[nodes], dx, shrink=1.0) <= 1.0:
                raise RuntimeError(
                    f"prescribed motion between t = {self.t:.6g} and t = {self.t + self.dt:.6g} carries a "
                    f"node into obstacle {k} ({type(o).__name__}); use a smaller dt or a path that avoids it"
                )

    # ------------------------------------------------------------ potential
    def energy(self, x: np.ndarray) -> float:
        e_contact = self.contact.energy(x)
        if not np.isfinite(e_contact):
            return np.inf
        if self.quasistatic:
            e_inertia = -float(np.sum(self.mass[:, None] * self.gravity * x))
        else:
            d = x - self._x_tilde
            e_inertia = 0.5 / self.dt**2 * float(np.sum(self.mass[:, None] * d * d))
        return e_inertia + self.elastic.energy(x) + e_contact + self.friction.energy(x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        g = self.elastic.gradient(x).reshape(-1, 3) + self.contact.gradient(x) + self.friction.gradient(x)
        if self.quasistatic:
            g -= self.mass[:, None] * self.gravity
        else:
            g += (self.mass / self.dt**2)[:, None] * (x - self._x_tilde)
        return g

    def hessian(self, x: np.ndarray):
        nb = self.contact.hessian_blocks(x, self.n) + self.friction.hessian_blocks(x, self.n)
        if self.quasistatic:
            if self.regularization > 0.0:
                nb += self.regularization * np.eye(3)
        else:
            nb += (self.mass / self.dt**2)[:, None, None] * np.eye(3)
        return self.assembler.assemble(self.elastic.hessian_blocks(x, self.project), nb)

    # ---------------------------------------------------------------- solve
    @staticmethod
    def _solve(H, b):
        A = H.tocsc()
        try:
            lu = spla.splu(A, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options={"SymmetricMode": True})
            return lu.solve(b)
        except RuntimeError:
            return spla.spsolve(A, b)

    def _newton(self, x: np.ndarray, info: StepInfo) -> np.ndarray:
        E = self.energy(x)
        if not np.isfinite(E):
            raise RuntimeError("state penetrates an obstacle; reduce dt or move bodies apart")
        free = ~self.fixed
        for it in range(1, self.max_newton + 1):
            g = self.gradient(x)
            g[self.fixed] = 0.0
            p = self._solve(self.hessian(x), -g.ravel()).reshape(-1, 3)
            p[self.fixed] = 0.0
            pinf = float(np.abs(p[free]).max()) if free.any() else 0.0
            info.residuals.append(pinf)
            info.iterations = it
            if pinf < self.newton_tol:
                info.converged = True
                break
            alpha = min(1.0, self.contact.max_step(x, p))
            gp = float(g.ravel() @ p.ravel())
            # Armijo backtracking; the round-off allowance keeps the search from
            # stalling once the energy decrease is below machine precision.
            slack = 1e-12 * (abs(E) + 1.0)
            while True:
                xt = x + alpha * p
                Et = self.energy(xt)
                if Et <= E + 1e-4 * alpha * gp + slack:
                    break
                alpha *= 0.5
                if alpha < 1e-10:
                    break
            if alpha < 1e-10 or not np.isfinite(Et):
                break  # line search stalled: keep the last accepted iterate
            info.alphas.append(alpha)
            x, E = xt, Et
        return x

    def step(self) -> StepInfo:
        """Advance the simulation by one time step (or one load step)."""
        t0 = time.perf_counter()
        info = StepInfo()
        h = self.dt
        x_n = self.x.copy()
        x = x_n.copy()
        self._apply_constraints(x, self.t + h)
        self._check_prescribed_path(x_n, x)
        if not self.quasistatic:
            self._x_tilde = x_n + h * self.v + h * h * self.gravity
            guess = np.where(self.fixed, 0.0, h * self.v)
            x = x + min(1.0, self.contact.max_step(x, guess)) * guess
        self.friction.update(self.contact, x_n, self.eps_v * h)
        x = self._newton(x, info)
        self.v = np.zeros_like(x) if self.quasistatic else (x - x_n) / h
        self.x = x
        self.t += h
        info.seconds = time.perf_counter() - t0
        self.history.append(info)
        return info

    # ------------------------------------------------------------ queries
    def kinetic_energy(self) -> float:
        return 0.5 * float(np.sum(self.mass[:, None] * self.v**2))

    def center_of_mass(self, nodes=None) -> np.ndarray:
        idx = slice(None) if nodes is None else nodes
        m = self.mass[idx]
        return (m[:, None] * self.x[idx]).sum(0) / m.sum()

    def contact_forces(self) -> list[np.ndarray]:
        """Total force (3-vector) that every obstacle exerts on the bodies."""
        xs = self.x[self.contact.nodes]
        out = []
        for o, lam in zip(self.contact.obstacles, self.contact.normal_forces(self.x)):
            out.append((lam[:, None] * o.normal(xs)).sum(0))
        return out

    def friction_force(self) -> np.ndarray:
        """Total friction force (3-vector) the obstacles exert on the bodies."""
        return -self.friction.gradient(self.x).sum(axis=0)

    def vertex_von_mises(self) -> np.ndarray:
        return element_to_vertex(self.mesh, self.elastic.von_mises(self.x))
