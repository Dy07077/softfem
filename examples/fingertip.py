"""A soft fingertip presses on a plate, slides across it and lifts off.

The fingertip pad is a soft ellipsoid around a rigid, dorsally placed bone
(the distal phalanx). The bone is driven kinematically; the pad deforms and
develops a contact patch. When the finger starts to slide, the pad first
shears while slip spreads from the rim of the patch inwards (incipient slip),
until the tangential force saturates at mu * F_n and the whole patch slides,
as Coulomb friction predicts.

    python examples/fingertip.py           # writes assets/fingertip.gif
    python examples/fingertip.py --quick   # fast low-resolution smoke run
"""
from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np
from scipy.spatial.transform import Rotation

import softfem as sf
from softfem.render import INK, Renderer, save_gif, stress_cmap

ASSETS = pathlib.Path(__file__).resolve().parents[1] / "assets"
MUTED, GRID = "#8b949e", "#30363d"
MU = 0.5
SEMI = np.array([0.0085, 0.012, 0.0075])  # pad semi-axes (m)
TILT = Rotation.from_euler("x", -20, degrees=True).as_matrix()
DURATION = 3.8


def smooth(s):
    s = np.clip(s, 0.0, 1.0)
    return s * s * (3 - 2 * s)


def finger_offset(t):
    """Rigid displacement of the bone: press, drag 16 mm towards the palm, lift (metres)."""
    press = -0.0032 * smooth(t / 0.8) + 0.0045 * smooth((t - 3.1) / 0.5)
    slide = -0.016 * smooth((t - 1.0) / 1.9)
    return np.array([0.0, slide, press])


def build(res):
    mesh = sf.ellipsoid(SEMI, res)
    local = mesh.X.copy()
    bone = ((local[:, 0] / 0.0045) ** 2 + ((local[:, 1] + 0.002) / 0.0085) ** 2 + ((local[:, 2] - 0.0026) / 0.0033) ** 2) <= 1.0
    rigid = bone | (local[:, 1] < -0.0105)  # bone + cut towards the middle phalanx
    X = local @ TILT.T
    shift = -(X[:, 2].min() - 0.0004)  # lowest point 0.4 mm above the plate
    X[:, 2] += shift
    return sf.TetMesh(X, mesh.T), np.where(rigid)[0], shift


def nail_mesh(shift, n=12):
    """A thin curved nail plate on the dorsal, distal part of the pad."""
    u = np.linspace(-0.55, 0.55, n)
    v = np.linspace(-0.2, 0.72, n)
    U, V = np.meshgrid(u, v, indexing="ij")
    x, y = U * SEMI[0], V * SEMI[1]
    z = SEMI[2] * np.sqrt(np.clip(1 - U**2 - V**2, 0.02, None)) + 0.00035
    P = np.stack([x, y, z], -1).reshape(-1, 3) @ TILT.T
    P[:, 2] += shift
    F = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j + 1
            F += [[a, b, c], [a, c, d]]
    return P, np.array(F)


def middle_phalanx_mesh(shift, n_around=24, n_len=14):
    """Rigid (not simulated) cylinder continuing the finger proximally, for context."""
    r = 0.0072
    y = np.linspace(-0.006, -0.03, n_len)
    phi = np.linspace(0, 2 * np.pi, n_around, endpoint=False)
    P = np.stack([np.tile(r * np.cos(phi), n_len), np.repeat(y, n_around), np.tile(r * np.sin(phi), n_len) + 0.0006], 1)
    P = P @ TILT.T
    P[:, 2] += shift
    F = []
    for i in range(n_len - 1):
        for j in range(n_around):
            a, b = i * n_around + j, i * n_around + (j + 1) % n_around
            c, d = a + n_around, b + n_around
            F += [[a, d, c], [a, b, d]]  # outward winding
    end = (n_len - 1) * n_around
    cap = np.array([[0.0, y[-1], 0.0006]]) @ TILT.T
    cap[:, 2] += shift
    F += [[len(P), end + j, end + (j + 1) % n_around] for j in range(n_around)]  # close the far end
    return np.concatenate([P, cap]), np.array(F)


def main(quick: bool = False) -> None:
    res = 6 if quick else 12
    mesh, rigid, shift = build(res)
    sim = sf.Simulator(
        mesh,
        sf.StableNeoHookean(8e4, 0.45),
        density=1100.0,
        dt=0.01,
        gravity=(0.0, 0.0, 0.0),
        obstacles=[sf.Plane(mu=MU)],
        dhat=1e-4,
        eps_v=2e-4,  # slip below 0.2 mm/s counts as sticking
    )
    X0 = mesh.X[rigid].copy()
    sim.fix(rigid, lambda t: X0 + finger_offset(t))
    faces = mesh.boundary_faces()
    nail_V, nail_F = nail_mesh(shift)
    mid_V, mid_F = middle_phalanx_mesh(shift)
    cmap = stress_cmap()

    hist_t, hist_n, hist_f = [], [], []
    r = Renderer((-0.016, 0.016), (-0.04, 0.02), (0.0, 0.026), size=(820, 400), tile=0.004, zoom=1.75,
                 elev=20, azim=-14, rect=(0.0, 0.0, 0.62, 1.0), focal_length=0.5)
    state = {"pmax": 4e4, "fmax": 0.6, "offset": None}

    def overlay(fig):
        fig.text(0.025, 0.92, "Soft fingertip: press, slide, lift", color=INK, fontsize=12.5, fontweight="bold")
        fig.text(0.025, 0.865, f"rigid bone + Stable Neo-Hookean pulp (E = 80 kPa), Coulomb friction μ = {MU}",
                 color=MUTED, fontsize=8)
        fig.text(0.025, 0.05, "colour: von Mises stress", color=MUTED, fontsize=8)
        ax = fig.add_axes((0.665, 0.53, 0.3, 0.36), facecolor="#161b22")
        xs, p, slip = state["xy"], state["p"], state["slip"]
        if len(p):
            ax.scatter(xs[:, 0] * 1e3, xs[:, 1] * 1e3, c=p / 1e3, cmap=cmap, vmin=0, vmax=state["pmax"] / 1e3, s=18, lw=0)
            ax.scatter(xs[slip, 0] * 1e3, xs[slip, 1] * 1e3, s=34, facecolors="none", edgecolors="white", lw=0.6)
        ax.set_xlim(-8, 8)
        ax.set_ylim(-8, 8)
        ax.set_aspect("equal")
        ax.set_title("contact patch, finger frame (mm)   ○ slipping", color=INK, fontsize=8, pad=3)
        _style(ax)
        ax = fig.add_axes((0.69, 0.1, 0.28, 0.3), facecolor="#161b22")
        ax.plot(hist_t, hist_n, color="#58a6ff", lw=1.4, label="normal $F_n$")
        ax.plot(hist_t, hist_f, color="#ffa657", lw=1.4, label="tangential $F_t$")
        ax.plot(hist_t, MU * np.asarray(hist_n), color="#ffa657", lw=0.8, ls="--", alpha=0.7, label="$\\mu F_n$")
        ax.set_xlim(0, DURATION)
        ax.set_ylim(0, state["fmax"] * 1.12)
        ax.set_xlabel("time (s)", color=MUTED, fontsize=7.5, labelpad=1)
        ax.set_ylabel("force (N)", color=MUTED, fontsize=7.5, labelpad=1)
        ax.legend(loc="upper right", fontsize=6.5, frameon=False, labelcolor=INK, ncol=3, handlelength=1.2, columnspacing=0.8)
        _style(ax)

    r.overlays = [overlay]
    frames, t0 = [], time.perf_counter()
    eps = sim.eps_v * sim.dt
    n_steps = 45 if quick else int(round(DURATION / sim.dt))
    for k in range(n_steps):
        sim.step()
        hist_t.append(sim.t)
        hist_n.append(sim.contact_forces()[0][2])
        hist_f.append(np.linalg.norm(sim.friction_force()[:2]))
        if k % 4 == 3 or quick:
            lam = sim.contact.normal_forces(sim.x)[0]
            on = lam > 1e-6 * max(lam.max(), 1e-30)
            nodes = sim.contact.nodes[on]
            centre = (X0 + finger_offset(sim.t)).mean(0)
            if state["offset"] is None and sim.t >= 0.9 and on.any():
                state["offset"] = sim.x[nodes, :2].mean(0) - centre[:2]
            off = state["offset"] if state["offset"] is not None else np.zeros(2)
            moved = np.zeros(len(nodes), dtype=bool)
            for idx, _, T, xr in sim.friction.groups:
                pos = {n: i for i, n in enumerate(idx)}
                for m, n in enumerate(nodes):
                    if n in pos:
                        i = pos[n]
                        moved[m] = np.linalg.norm(T[i].T @ (sim.x[n] - xr[i])) > eps
            p = lam[on] / sim.contact.weights[on]
            state.update(xy=sim.x[nodes, :2] - centre[:2] - off, p=p, slip=moved)
            if len(p):
                state["pmax"] = max(state["pmax"], np.percentile(p, 98))
            state["fmax"] = max(state["fmax"], max(hist_n))
            vm = sim.vertex_von_mises()
            off = finger_offset(sim.t)
            frames.append(r.render([
                {"x": sim.x, "faces": faces, "scalars": vm, "clim": (0.0, 3.5e4), "cmap": cmap, "edges": 0.25},
                {"x": nail_V + off, "faces": nail_F, "color": "#f2d7cf", "edges": 0.0},
                {"x": mid_V + off, "faces": mid_F, "color": "#3d4b63", "edges": 0.15},
            ]))
    iters = np.mean([h.iterations for h in sim.history])
    print(f"{len(sim.history)} steps, {iters:.1f} Newton iterations/step, {time.perf_counter() - t0:.0f} s")
    if not quick:
        k = int(2.6 / sim.dt)
        print(f"peak normal force {max(hist_n):.3f} N; F_t/F_n at t = 2.6 s: {hist_f[k] / hist_n[k]:.3f} (mu = {MU})")
    ASSETS.mkdir(exist_ok=True)
    out = ASSETS / ("fingertip_quick.gif" if quick else "fingertip.gif")
    save_gif(frames, out, fps=25)
    print("wrote", out)


def _style(ax):
    ax.tick_params(colors=MUTED, labelsize=6.5, length=2, pad=1)
    for s in ax.spines.values():
        s.set_color(GRID)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
