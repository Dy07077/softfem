"""A soft bar clamped at one end and twisted by 1.25 turns at the other.

    python examples/twist.py           # writes assets/twist.gif
    python examples/twist.py --quick   # fast low-resolution smoke run
"""
from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np

import softfem as sf
from softfem.render import INK, Renderer, save_gif, stress_cmap

ASSETS = pathlib.Path(__file__).resolve().parents[1] / "assets"


def main(quick: bool = False) -> None:
    fps, substeps, seconds, turns = 30, 2, (0.5 if quick else 3.6), 1.25
    L, a = 0.5, 0.08
    res = (10, 2, 2) if quick else (36, 6, 6)
    bar = sf.box((L, a, a), res, origin=(-L / 2, -a / 2, 0.12))
    sim = sf.Simulator(bar, sf.StableNeoHookean(2e5, 0.4), density=1000.0, dt=1.0 / (fps * substeps))
    left = np.where(bar.X[:, 0] < -L / 2 + 1e-9)[0]
    right = np.where(bar.X[:, 0] > L / 2 - 1e-9)[0]
    axis_point = np.array([0.0, 0.0, 0.12 + a / 2])
    R0 = bar.X[right] - axis_point

    def twist(t):
        s = np.clip(t / 3.0, 0.0, 1.0)
        th = 2.0 * np.pi * turns * s * s * (3 - 2 * s)
        c, s_ = np.cos(th), np.sin(th)
        y = c * R0[:, 1] - s_ * R0[:, 2]
        z = s_ * R0[:, 1] + c * R0[:, 2]
        return np.stack([R0[:, 0], y, z], axis=1) + axis_point

    sim.fix(left)
    sim.fix(right, twist)
    faces = bar.boundary_faces()
    cmap = stress_cmap()
    r = Renderer((-0.32, 0.32), (-0.2, 0.2), (0.0, 0.3), size=(560, 400), tile=0.08, zoom=2.0, elev=20, azim=-70, ground=0.0)

    def overlay(fig):
        fig.text(0.03, 0.92, "Twist", color=INK, fontsize=13, fontweight="bold")
        fig.text(0.03, 0.865, "clamped bar under torsion, coloured by von Mises stress", color="#8b949e", fontsize=8.5)
        s = np.clip(sim.t / 3.0, 0, 1)
        fig.text(0.97, 0.92, f"{turns * s * s * (3 - 2 * s):4.2f} turns", color="#8b949e", fontsize=9, ha="right", family="monospace")

    r.overlays = [overlay]
    frames, t0 = [], time.perf_counter()
    for k in range(int(seconds * fps)):
        for _ in range(substeps):
            sim.step()
        vm = sim.vertex_von_mises()
        frames.append(r.render([{"x": sim.x, "faces": faces, "scalars": vm, "clim": (0.0, 2.0e5), "cmap": cmap, "edges": 0.25}]))
    iters = np.mean([h.iterations for h in sim.history])
    print(f"{len(sim.history)} steps, {iters:.1f} Newton iterations/step, {time.perf_counter() - t0:.0f} s")
    ASSETS.mkdir(exist_ok=True)
    out = ASSETS / ("twist_quick.gif" if quick else "twist.gif")
    save_gif(frames, out, fps=fps)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
