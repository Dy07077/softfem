"""A soft foam sheet drapes over a rigid ball and settles on the floor.

    python examples/drape.py           # writes assets/drape.gif
    python examples/drape.py --quick   # fast low-resolution smoke run
"""
from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np

import softfem as sf
from softfem.render import INK, PALETTE, Renderer, save_gif

ASSETS = pathlib.Path(__file__).resolve().parents[1] / "assets"


def main(quick: bool = False) -> None:
    n = 10 if quick else 26
    fps, substeps, seconds = 30, 5, (0.5 if quick else 2.4)
    sheet = sf.box((0.46, 0.46, 0.012), (n, n, 1), origin=(-0.23, -0.23, 0.25))
    ball = sf.Sphere((0.0, 0.0, 0.1), 0.1, mu=0.5)
    sim = sf.Simulator(
        sheet,
        sf.StableNeoHookean(1.2e4, 0.3),
        density=400.0,
        dt=1.0 / (fps * substeps),
        obstacles=[sf.Plane(mu=0.5), ball],
        dhat=4e-4,
        eps_v=1e-2,
    )
    faces = sheet.boundary_faces()
    sphere = sf.ball(0.1 - 4e-4, 10, center=ball.center)
    r = Renderer((-0.34, 0.34), (-0.34, 0.34), (0.0, 0.3), size=(560, 400), tile=0.08, zoom=1.85, elev=24)

    def overlay(fig):
        fig.text(0.03, 0.92, "Drape", color=INK, fontsize=13, fontweight="bold")
        fig.text(0.03, 0.865, "soft sheet on a rigid ball, frictional contact", color="#8b949e", fontsize=8.5)
        fig.text(0.97, 0.92, f"t = {sim.t:4.2f} s", color="#8b949e", fontsize=9, ha="right", family="monospace")

    r.overlays = [overlay]
    frames, t0 = [], time.perf_counter()
    n_frames = int(seconds * fps)
    for k in range(n_frames):
        for _ in range(substeps):
            sim.step()
        r.azim = -60.0 + 25.0 * k / max(n_frames - 1, 1)
        frames.append(
            r.render(
                [
                    {"x": sphere.X, "faces": sphere.boundary_faces(), "color": "#6e7681"},
                    {"x": sim.x, "faces": faces, "color": PALETTE[4], "edges": 0.3},
                ]
            )
        )
    iters = np.mean([h.iterations for h in sim.history])
    print(f"{len(sim.history)} steps, {iters:.1f} Newton iterations/step, min gap {sim.contact.min_distance(sim.x):.2e}, "
          f"{time.perf_counter() - t0:.0f} s")
    ASSETS.mkdir(exist_ok=True)
    out = ASSETS / ("drape_quick.gif" if quick else "drape.gif")
    save_gif(frames, out, fps=fps)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
