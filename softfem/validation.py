"""Reproducible benchmark problems with closed-form reference solutions.

Each function sets up a simulation, runs it and returns the measured
quantities next to the analytic prediction, so the same code drives the unit
tests (coarse meshes) and the figures in the README (fine meshes).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contact import Plane
from .fem import nodal_areas
from .materials import LinearElastic, StableNeoHookean
from .mesh import ball_octant, box
from .solver import Simulator

__all__ = ["HertzResult", "hertz_contact", "cantilever_tip_deflection", "incline_acceleration"]


@dataclass
class HertzResult:
    delta: np.ndarray  # approach of the ball centre towards the plane
    force: np.ndarray  # FEM contact force
    force_hertz: np.ndarray
    radius: np.ndarray  # FEM contact radius
    radius_hertz: np.ndarray
    pressure_r: np.ndarray  # radial position of contact nodes (last load step)
    pressure: np.ndarray  # nodal contact pressure (last load step)
    n_nodes: int
    n_tets: int
    newton_iterations: list


def hertz_contact(res=8, grading=1.6, delta_max=0.04, n_steps=8, E=1.0, nu=0.3, radius=1.0, dhat=1e-4) -> HertzResult:
    """Elastic ball pressed onto a rigid frictionless plane (Hertz, 1882).

    Hertz: ``F = 4/3 E* sqrt(R) delta^(3/2)``, ``a = sqrt(R delta)``,
    ``p(r) = p0 sqrt(1 - r^2/a^2)`` with ``E* = E / (1 - nu^2)``.

    Only one octant is simulated: symmetry planes x = 0 and y = 0 are rollers
    and the equator moves down rigidly by ``delta`` (a ball squeezed between
    two frictionless plates). Forces are multiplied by 4 for the full patch.
    """
    R = float(radius)
    mesh = ball_octant(R, res, grading)
    X = mesh.X
    faces = mesh.boundary_faces()
    on_sphere = np.abs(np.linalg.norm(X, axis=1) - R) < 1e-9 * R
    sph = faces[on_sphere[faces].all(axis=1)]
    nodes = np.unique(sph)
    weights = nodal_areas(X, sph, mesh.n_nodes)[nodes]
    gap0 = dhat * R
    sim = Simulator(
        mesh,
        StableNeoHookean(E, nu),
        quasistatic=True,
        gravity=(0.0, 0.0, 0.0),
        obstacles=[Plane(point=(0.0, 0.0, -R - gap0))],
        dhat=dhat * R,
        contact_nodes=nodes,
        contact_weights=weights,
        newton_tol=1e-9,
        dt=1.0 / n_steps,
    )
    tol = 1e-12 * R
    for axis in (0, 1):
        on_plane = np.where(np.abs(X[:, axis]) < tol)[0]
        sim.fix(on_plane, np.zeros((len(on_plane), 3)), axes=(axis,))
    equator = np.where(np.abs(X[:, 2]) < tol)[0]

    def equator_target(t):
        p = np.zeros((len(equator), 3))
        p[:, 2] = -(gap0 + delta_max * R * t)
        return p

    sim.fix(equator, equator_target, axes=(2,))
    Es = E / (1.0 - nu**2)
    delta, force, rad, its = [], [], [], []
    for _ in range(n_steps):
        info = sim.step()
        its.append(info.iterations)
        lam = sim.contact.normal_forces(sim.x)[0]
        r_nodes = np.linalg.norm(sim.x[nodes, :2], axis=1)
        delta.append(delta_max * R * sim.t)
        force.append(4.0 * lam.sum())
        rad.append(r_nodes[lam > 1e-3 * lam.max()].max() if lam.max() > 0 else 0.0)
    delta = np.array(delta)
    pressure = lam / weights
    return HertzResult(
        delta=delta,
        force=np.array(force),
        force_hertz=4.0 / 3.0 * Es * np.sqrt(R) * delta**1.5,
        radius=np.array(rad),
        radius_hertz=np.sqrt(R * delta),
        pressure_r=r_nodes,
        pressure=pressure,
        n_nodes=mesh.n_nodes,
        n_tets=mesh.n_tets,
        newton_iterations=its,
    )


def cantilever_tip_deflection(n=2, length=10.0, E=1e9, density=1000.0, g=9.81):
    """Tip deflection of a square cantilever (``length x 1 x 1``) under gravity.

    Returns ``(fem, euler_bernoulli, timoshenko, n_dof)``. Linear elasticity
    with ``nu = 0`` matches the assumptions of beam theory.
    """
    W = H = 1.0
    q = density * g * W * H
    I = W * H**3 / 12.0
    eb = q * length**4 / (8.0 * E * I)
    timo = eb + q * length**2 / (2.0 * (5.0 / 6.0) * (E / 2.0) * W * H)
    mesh = box((length, W, H), (10 * n, n, n))
    sim = Simulator(mesh, LinearElastic(E, 0.0), density=density, quasistatic=True, gravity=(0.0, 0.0, -g), newton_tol=1e-12)
    sim.fix(np.where(mesh.X[:, 0] < 1e-9)[0])
    sim.step()
    tip = mesh.X[:, 0] > length - 1e-9
    fem = -float((sim.x[tip, 2] - mesh.X[tip, 2]).mean())
    return fem, eb, timo, 3 * mesh.n_nodes


def incline_acceleration(theta_deg, mu=0.3, duration=1.0, dt=0.01, res=4, size=0.1):
    """Soft cube on a plane inclined by ``theta``; returns ``(a_fem, a_coulomb, slip)``.

    Coulomb: the cube sticks if ``tan(theta) <= mu``, otherwise it slides with
    ``a = g (sin(theta) - mu cos(theta))``. Gravity is rotated instead of the plane.
    """
    th = np.radians(theta_deg)
    mesh = box((size,) * 3, (res,) * 3, origin=(-size / 2, -size / 2, 0.0))
    dhat = 1e-3 * mesh.bbox_diagonal()
    mesh = mesh.transformed(translation=(0.0, 0.0, 0.5 * dhat))
    g = 9.81
    sim = Simulator(
        mesh,
        StableNeoHookean(1e5, 0.3),
        dt=dt,
        gravity=(g * np.sin(th), 0.0, -g * np.cos(th)),
        obstacles=[Plane(mu=mu)],
        dhat=dhat,
    )
    ts, xs = [], []
    for _ in range(int(round(duration / dt))):
        sim.step()
        ts.append(sim.t)
        xs.append(sim.center_of_mass()[0])
    ts, xs = np.array(ts), np.array(xs)
    late = ts >= 0.3 * duration
    a_fem = 2.0 * np.polyfit(ts[late], xs[late], 2)[0]
    a_coulomb = max(0.0, g * (np.sin(th) - mu * np.cos(th)))
    return float(a_fem), float(a_coulomb), float(xs[-1] - xs[0])
