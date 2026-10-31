"""Three jellies of different stiffness dropped onto a frictional floor.

    python examples/jelly_drop.py           # writes assets/jelly_drop.gif
    python examples/jelly_drop.py --quick   # fast low-resolution smoke run
"""
from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np
from scipy.spatial.transform import Rotation

import softfem as sf
from softfem.render import INK, PALETTE, Renderer, save_gif

ASSETS = pathlib.Path(__file__).resolve().parents[1] / "assets"
MUTED = "#8b949e"


def build(res: int):
    rot = lambda *deg: Rotation.from_euler("xyz", deg, degrees=True).as_matrix()  # noqa: E731
    cube = sf.box((0.15,) * 3, (res,) * 3, origin=(-0.075,) * 3)
    cube = cube.transformed(rotation=rot(30, 40, 15), translation=(-0.30, 0.03, 0.48))
    ball = sf.ball(0.085, res + 2, center=(0.0, -0.03, 0.62))
    bean = sf.ellipsoid((0.12, 0.065, 0.065), res + 2).transformed(rotation=rot(0, 25, 30), translation=(0.30, 0.0, 0.40))
    return [cube, ball, bean], [8e3, 3e4, 1.2e5]


def main(quick: bool = False) -> None:
    fps, seconds = 30, (0.6 if quick else 3.0)
    substeps = 3 if quick else 6
    bodies, stiffness = build(3 if quick else 6)
    mesh = sf.merge(bodies)
    E = np.concatenate([np.full(b.n_tets, e) for b, e in zip(bodies, stiffness)])
    sim = sf.Simulator(
        mesh,
        sf.StableNeoHookean(E, 0.4),
        density=1000.0,
        dt=1.0 / (fps * substeps),
        obstacles=[sf.Plane(mu=0.6)],
        dhat=5e-4,
        eps_v=1e-2,  # 1 cm/s stick threshold: invisible here, and Newton converges ~2x faster
    )
    faces = mesh.boundary_faces()
    offsets = np.cumsum([0] + [b.n_nodes for b in bodies])
    groups = [faces[(faces.min(1) >= a) & (faces.max(1) < b)] for a, b in zip(offsets[:-1], offsets[1:])]

    r = Renderer((-0.5, 0.5), (-0.32, 0.32), (0.0, 0.5), size=(720, 405), tile=0.1, zoom=2.0, elev=16)

    def overlay(fig):
        fig.text(0.03, 0.92, "softfem", color=INK, fontsize=14, fontweight="bold")
        fig.text(0.03, 0.865, "implicit FEM  ·  Stable Neo-Hookean  ·  IPC barrier contact + friction", color=MUTED, fontsize=8.5)
        for i, (c, e) in enumerate(zip(PALETTE, stiffness)):
            fig.text(0.03 + 0.14 * i, 0.055, "●", color=c, fontsize=12, va="center")
            fig.text(0.057 + 0.14 * i, 0.055, f"E = {e / 1e3:g} kPa", color=INK, fontsize=8.5, va="center")
        fig.text(0.97, 0.92, f"t = {sim.t:4.2f} s", color=MUTED, fontsize=9, ha="right", family="monospace")

    r.overlays = [overlay]
    frames, t0 = [], time.perf_counter()
    n_frames = int(seconds * fps)
    for k in range(n_frames):
        for _ in range(substeps):
            sim.step()
        r.azim = -72.0 + 20.0 * k / max(n_frames - 1, 1)
        frames.append(r.render([{"x": sim.x, "faces": f, "color": c, "edges": 0.28} for f, c in zip(groups, PALETTE)]))
    iters = np.mean([h.iterations for h in sim.history])
    print(f"{len(sim.history)} steps, {iters:.1f} Newton iterations/step, {time.perf_counter() - t0:.0f} s")
    ASSETS.mkdir(exist_ok=True)
    out = ASSETS / ("jelly_drop_quick.gif" if quick else "jelly_drop.gif")
    save_gif(frames, out, fps=fps)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
