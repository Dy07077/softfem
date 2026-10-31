"""Offline rendering of surface meshes with matplotlib (for figures and GIFs).

The renderer draws a shaded, depth-sorted triangle soup with back-face
culling, a tiled ground plane and a projected drop shadow. It uses the Agg
canvas directly, so it never touches the global matplotlib backend.
"""
from __future__ import annotations

import numpy as np

__all__ = ["Renderer", "save_gif", "stress_cmap", "SKY", "INK", "PALETTE"]

SKY = "#0d1117"  # GitHub dark background
INK = "#e6edf3"
PALETTE = ["#58a6ff", "#f778ba", "#3fb950", "#ffa657", "#bc8cff", "#39c5cf"]


def stress_cmap():
    """Bright sequential colormap (blue -> violet -> pink -> amber -> cream)."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "softfem", ["#1f6feb", "#8957e5", "#db61a2", "#f0883e", "#ffdf5d", "#fffbe6"]
    )


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


class Renderer:
    """Render one or more triangle meshes into an RGB array.

    Args:
        xlim, ylim, zlim: scene bounds (kept at equal aspect).
        size: output size in pixels.
        elev, azim: camera angles in degrees (matplotlib convention).
        ground: height of the ground plane, or ``None`` for no ground.
        tile: edge length of the ground tiles.
    """

    def __init__(
        self,
        xlim,
        ylim,
        zlim,
        size=(640, 400),
        elev=22.0,
        azim=-60.0,
        ground: float | None = 0.0,
        tile: float = 0.1,
        background: str = SKY,
        ground_colors=("#222a37", "#2a3342"),
        shadow_color="#141922",
        light=(-0.3, -0.5, 1.0),
        focal_length: float = 0.6,
        zoom: float = 1.5,
        rect=(0.0, 0.0, 1.0, 1.0),
        dpi: int = 100,
    ):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        self.lims = (tuple(xlim), tuple(ylim), tuple(zlim))
        self.elev, self.azim = float(elev), float(azim)
        self.ground = ground
        self.tile = float(tile)
        self.background = background
        self.ground_colors = ground_colors
        self.shadow_color = shadow_color
        self.light = _unit(light)
        self.focal_length = focal_length
        self.zoom = float(zoom)
        self.rect = tuple(rect)
        self.fig = Figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi, facecolor=background)
        self.canvas = FigureCanvasAgg(self.fig)
        self.overlays = []  # callables (fig) -> None drawn on every frame

    # ----------------------------------------------------------------- camera
    def eye_direction(self) -> np.ndarray:
        e, a = np.radians(self.elev), np.radians(self.azim)
        return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])

    def _axes(self):
        ax = self.fig.add_axes(self.rect, projection="3d", facecolor=self.background)
        ax.set_proj_type("persp", focal_length=self.focal_length)
        ax.computed_zorder = False
        (x0, x1), (y0, y1), (z0, z1) = self.lims
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_zlim(z0, z1)
        ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0), zoom=self.zoom)
        ax.view_init(self.elev, self.azim)
        ax.set_axis_off()
        return ax

    # ------------------------------------------------------------------ draw
    def _ground(self, ax):
        """Checkerboard that extends past the view and fades into the background."""
        from matplotlib.colors import to_rgb
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        (x0, x1), (y0, y1), _ = self.lims
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        span = max(x1 - x0, y1 - y0)
        t = self.tile
        n = int(np.ceil(1.6 * span / t))
        xs = np.floor(cx / t) * t + t * np.arange(-n, n + 1)
        ys = np.floor(cy / t) * t + t * np.arange(-n, n + 1)
        bg = np.array(to_rgb(self.background))
        tiles = [np.array(to_rgb(c)) for c in self.ground_colors]
        quads, cols = [], []
        z = self.ground
        for i in range(len(xs) - 1):
            for j in range(len(ys) - 1):
                xa, xb, ya, yb = xs[i], xs[i + 1], ys[j], ys[j + 1]
                r = np.hypot(0.5 * (xa + xb) - cx, 0.5 * (ya + yb) - cy) / span
                fade = np.clip((r - 0.45) / 0.9, 0.0, 1.0) ** 0.8
                quads.append([(xa, ya, z), (xb, ya, z), (xb, yb, z), (xa, yb, z)])
                cols.append(tiles[(i + j) % 2] * (1 - fade) + bg * fade)
        pc = Poly3DCollection(quads, facecolors=cols, edgecolors="none", zorder=0)
        pc.set_clip_on(False)
        ax.add_collection3d(pc)

    def _shadow(self, ax, tris):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        L = self.light
        if L[2] <= 1e-3:
            return
        h = tris[..., 2] - self.ground
        proj = tris - (h / L[2])[..., None] * L
        proj[..., 2] = self.ground + 1e-6
        pc = Poly3DCollection(proj, facecolors=self.shadow_color, edgecolors="none", zorder=1)
        pc.set_clip_on(False)
        ax.add_collection3d(pc)

    def _shade(self, tris, base_rgb):
        """Key light (fixed in the world, casts the shadow) + camera fill + specular."""
        n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
        eye = self.eye_direction()
        key = np.clip(n @ self.light, 0.0, 1.0)
        fill = np.clip(n @ eye, 0.0, 1.0)
        half = _unit(self.light + eye)
        spec = np.clip(n @ half, 0.0, 1.0) ** 48
        k = 0.30 + 0.50 * key + 0.30 * fill
        rgb = base_rgb * k[:, None] + 0.30 * spec[:, None]
        return np.clip(rgb, 0.0, 1.0), n

    def render(self, meshes) -> np.ndarray:
        """Draw meshes and return an ``(H, W, 3)`` uint8 image.

        ``meshes`` is a list of dicts with keys ``x`` (vertices), ``faces`` and
        either ``color`` (RGB tuple / hex) or ``scalars`` (per-vertex values)
        with optional ``cmap`` and ``clim``. ``edges`` (float alpha) draws the
        mesh wireframe.
        """
        from matplotlib import colormaps
        from matplotlib.colors import to_rgb
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        self.fig.clear()
        ax = self._axes()
        all_tris, all_cols, all_edges = [], [], []
        for m in meshes:
            tris = np.asarray(m["x"])[np.asarray(m["faces"])]
            if "scalars" in m:
                s = np.asarray(m["scalars"])[np.asarray(m["faces"])].mean(axis=1)
                lo, hi = m.get("clim", (s.min(), s.max()))
                cmap = m.get("cmap", None) or stress_cmap()
                cmap = colormaps[cmap] if isinstance(cmap, str) else cmap
                base = cmap(np.clip((s - lo) / max(hi - lo, 1e-300), 0.0, 1.0))[:, :3]
            else:
                base = np.broadcast_to(np.array(to_rgb(m.get("color", "#58a6ff"))), (len(tris), 3))
            rgb, n = self._shade(tris, base)
            centers = tris.mean(axis=1)
            eye = np.mean(self.lims, axis=1) + 4.0 * np.ptp(np.array(self.lims), axis=1).max() * self.eye_direction()
            visible = np.einsum("ij,ij->i", n, eye - centers) > 0.0
            all_tris.append(tris[visible])
            all_cols.append(rgb[visible])
            a = float(m.get("edges", 0.0))
            ec = np.concatenate([rgb[visible] * 0.55, np.full((visible.sum(), 1), a)], axis=1)
            all_edges.append(ec)
        tris = np.concatenate(all_tris)
        if self.ground is not None:
            self._ground(ax)
            self._shadow(ax, np.concatenate([np.asarray(m["x"])[np.asarray(m["faces"])] for m in meshes]))
        pc = Poly3DCollection(
            tris,
            facecolors=np.concatenate(all_cols),
            edgecolors=np.concatenate(all_edges),
            linewidths=0.25,
            zorder=2,
        )
        pc.set_clip_on(False)
        ax.add_collection3d(pc)
        for draw in self.overlays:
            draw(self.fig)
        self.canvas.draw()
        return np.asarray(self.canvas.buffer_rgba())[..., :3].copy()


def save_gif(frames, path, fps: float = 30.0, colors: int = 192) -> None:
    """Write RGB frames to an optimised looping GIF with a shared palette."""
    from PIL import Image

    frames = [np.asarray(f, dtype=np.uint8) for f in frames]
    step = max(1, len(frames) // 12)
    sample = np.concatenate(frames[::step], axis=0)
    palette = Image.fromarray(sample).quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    imgs = [Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    imgs[0].save(
        path,
        save_all=True,
        append_images=imgs[1:],
        duration=int(round(1000.0 / fps)),
        loop=0,
        optimize=True,
        disposal=1,
    )
