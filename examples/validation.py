"""Reproduce the validation figure (assets/validation.png) and its numbers.

    python examples/validation.py           # ~4 minutes, fine meshes
    python examples/validation.py --quick   # coarse meshes, seconds
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from softfem.render import INK, PALETTE, SKY
from softfem.validation import cantilever_tip_deflection, hertz_contact, incline_acceleration

ASSETS = pathlib.Path(__file__).resolve().parents[1] / "assets"
MUTED, GRID, PANEL = "#8b949e", "#30363d", "#161b22"


def style(ax, title):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8, fontweight="bold")
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(color=GRID, lw=0.6, alpha=0.8)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def legend(ax, **kw):
    ax.legend(frameon=False, labelcolor=INK, fontsize=8.5, **kw)


def main(quick: bool = False) -> None:
    from matplotlib.figure import Figure

    t0 = time.perf_counter()
    results = {}
    hertz_res = (4, 6) if quick else (8, 12, 18)
    hertz = [hertz_contact(res=r, grading=1.6, n_steps=4 if quick else 8) for r in hertz_res]
    beams_n = (1, 2, 3) if quick else (1, 2, 3, 4, 6, 8)
    beams = [cantilever_tip_deflection(n) for n in beams_n]
    thetas = (10, 30) if quick else (2, 6, 10, 14, 16, 18, 20, 24, 28, 32, 36, 40)
    incline = [incline_acceleration(t, mu=0.3, duration=0.6 if quick else 1.0) for t in thetas]

    fig = Figure(figsize=(12, 7.8), dpi=110, facecolor=SKY)
    axs = fig.subplots(2, 2, gridspec_kw=dict(hspace=0.4, wspace=0.22, left=0.07, right=0.98, top=0.85, bottom=0.08))
    fig.text(0.07, 0.95, "softfem validation", color=INK, fontsize=15, fontweight="bold")
    fig.text(0.07, 0.915, "every curve is a closed-form reference; every marker is a simulation", color=MUTED, fontsize=9.5)

    # (a) Hertz force-indentation
    ax = axs[0, 0]
    finest = hertz[-1]
    d = np.geomspace(finest.delta[0], finest.delta[-1], 50)
    ax.loglog(d, 4 / 3 / (1 - 0.3**2) * d**1.5, color=INK, lw=1.4, label="Hertz  $F=\\frac{4}{3}E^*\\sqrt{R}\\,\\delta^{3/2}$")
    for h, c in zip(hertz, PALETTE[:3]):
        ax.loglog(h.delta, h.force, "o", ms=4.5, color=c, label=f"FEM, {h.n_tets:,} tets (octant)")
    slope = np.polyfit(np.log(finest.delta[1:]), np.log(finest.force[1:]), 1)[0]
    ratios = [float(h.force[-1] / h.force_hertz[-1]) for h in hertz]
    ax.text(0.97, 0.06, f"fitted slope {slope:.3f}\nF/F$_H$ = " + " → ".join(f"{r:.2f}" for r in ratios),
            transform=ax.transAxes, ha="right", va="bottom", color=MUTED, fontsize=8.5)
    ax.set_xlabel("indentation  $\\delta / R$")
    ax.set_ylabel("contact force  $F / (E R^2)$")
    style(ax, "Hertz contact: ball on rigid plane")
    legend(ax, loc="upper left")
    results["hertz"] = {"slope": float(slope), "force_ratio_by_mesh": dict(zip([h.n_tets for h in hertz], ratios)),
                        "radius_ratio": float(finest.radius[-1] / finest.radius_hertz[-1])}

    # (b) Hertz pressure distribution
    ax = axs[0, 1]
    a = finest.radius_hertz[-1]
    p0 = 3 * finest.force_hertz[-1] / (2 * np.pi * a * a)
    xx = np.linspace(0, 1, 200)
    ax.plot(xx, np.sqrt(1 - xx**2), color=INK, lw=1.4, label="Hertz  $p_0\\sqrt{1-r^2/a^2}$")
    m = finest.pressure > 0
    ax.plot(finest.pressure_r[m] / a, finest.pressure[m] / p0, "o", ms=3.5, color=PALETTE[2], alpha=0.85, label="FEM nodal pressure")
    ax.set_xlim(0, 1.5)
    ax.set_ylim(0, 1.25)
    ax.set_xlabel("radius  $r / a_{Hertz}$")
    ax.set_ylabel("pressure  $p / p_0$")
    style(ax, f"Contact pressure at $\\delta = {finest.delta[-1]:.2f}R$")
    legend(ax, loc="upper right")

    # (c) Cantilever
    ax = axs[1, 0]
    dofs = [b[3] for b in beams]
    ratio = [b[0] / b[1] for b in beams]
    ax.axhline(beams[0][2] / beams[0][1], color=INK, lw=1.0, ls="--", label="Timoshenko beam")
    ax.axhline(1.0, color=INK, lw=1.4, label="Euler–Bernoulli  $\\delta = qL^4/8EI$")
    ax.semilogx(dofs, ratio, "o-", ms=5, color=PALETTE[3], lw=1.2, label="FEM, linear tets")
    for n, x, y in zip(beams_n, dofs, ratio):
        ax.annotate(f"{n}×{n}", (x, y), textcoords="offset points", xytext=(0, -13), ha="center", color=MUTED, fontsize=7.5)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("degrees of freedom")
    ax.set_ylabel("tip deflection / beam theory")
    style(ax, "Cantilever under gravity (L/H = 10)")
    legend(ax, loc="lower right")
    results["cantilever"] = {f"{n}x{n}": float(r) for n, r in zip(beams_n, ratio)}

    # (d) Incline friction
    ax = axs[1, 1]
    th = np.linspace(0, 42, 300)
    mu = 0.3
    coul = np.maximum(0, 9.81 * (np.sin(np.radians(th)) - mu * np.cos(np.radians(th))))
    ax.plot(th, coul, color=INK, lw=1.4, label="Coulomb  $a = g(\\sin\\theta - \\mu\\cos\\theta)^+$")
    ax.plot(thetas, [r[0] for r in incline], "o", ms=5.5, color=PALETTE[1], label="FEM soft cube, $\\mu = 0.3$")
    ax.axvline(np.degrees(np.arctan(mu)), color=MUTED, lw=0.8, ls=":")
    ax.text(np.degrees(np.arctan(mu)) + 0.6, 3.6, "$\\tan^{-1}\\mu$", color=MUTED, fontsize=9)
    ax.set_xlabel("incline angle  $\\theta$ (deg)")
    ax.set_ylabel("sliding acceleration (m/s$^2$)")
    style(ax, "Stick–slip on an incline")
    legend(ax, loc="upper left")
    err = max(abs(r[0] - r[1]) for r in incline)
    results["incline"] = {"max_abs_error_m_s2": float(err), "points": [(t, r[0], r[1]) for t, r in zip(thetas, incline)]}

    ASSETS.mkdir(exist_ok=True)
    out = ASSETS / ("validation_quick.png" if quick else "validation.png")
    fig.savefig(out, facecolor=SKY)
    (ASSETS / ("validation_quick.json" if quick else "validation.json")).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"wrote {out} in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
