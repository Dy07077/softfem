"""Tetrahedral meshes: generation, orientation and surface extraction."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["TetMesh", "signed_volumes", "grid", "box", "ball", "ellipsoid", "ball_octant", "merge"]

# Kuhn (Freudenthal) split of the unit cube into 6 tetrahedra that share the
# diagonal 000-111. Corner id = i + 2j + 4k. Neighbouring cubes split their
# shared face along the same diagonal, so the resulting mesh is conforming.
_KUHN = np.array(
    [[0, 1, 3, 7], [0, 1, 5, 7], [0, 2, 3, 7], [0, 2, 6, 7], [0, 4, 5, 7], [0, 4, 6, 7]]
)

# Faces of a positively oriented tet (a, b, c, d), wound with outward normals.
_TET_FACES = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])


def signed_volumes(X: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Signed volume of every tetrahedron ``T`` with vertex positions ``X``."""
    a, b, c, d = (X[T[:, k]] for k in range(4))
    return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0


@dataclass
class TetMesh:
    """A linear tetrahedral mesh.

    Attributes:
        X: ``(n, 3)`` rest positions.
        T: ``(m, 4)`` vertex indices, re-ordered on construction so that every
            element has positive volume.
    """

    X: np.ndarray
    T: np.ndarray

    def __post_init__(self) -> None:
        self.X = np.ascontiguousarray(self.X, dtype=float)
        self.T = np.ascontiguousarray(self.T, dtype=np.int64)
        vol = signed_volumes(self.X, self.T)
        flip = vol < 0
        if np.any(flip):
            self.T[flip] = self.T[flip][:, [0, 1, 3, 2]]

    @property
    def n_nodes(self) -> int:
        return len(self.X)

    @property
    def n_tets(self) -> int:
        return len(self.T)

    def volumes(self) -> np.ndarray:
        return signed_volumes(self.X, self.T)

    def boundary_faces(self) -> np.ndarray:
        """Triangles that belong to exactly one tetrahedron, wound outward."""
        faces = self.T[:, _TET_FACES].reshape(-1, 3)
        s = np.sort(faces, axis=1)
        n = np.int64(self.n_nodes)
        key = (s[:, 0] * n + s[:, 1]) * n + s[:, 2]
        _, first, counts = np.unique(key, return_index=True, return_counts=True)
        return faces[first[counts == 1]]

    def surface_nodes(self) -> np.ndarray:
        return np.unique(self.boundary_faces())

    def bbox_diagonal(self) -> float:
        return float(np.linalg.norm(self.X.max(0) - self.X.min(0)))

    def transformed(self, *, scale=1.0, rotation=None, translation=(0.0, 0.0, 0.0)) -> "TetMesh":
        """Return a copy scaled, then rotated (3x3 matrix), then translated."""
        X = self.X * np.asarray(scale, dtype=float)
        if rotation is not None:
            X = X @ np.asarray(rotation, dtype=float).T
        return TetMesh(X + np.asarray(translation, dtype=float), self.T.copy())


def grid(xs, ys, zs) -> TetMesh:
    """Tet mesh of the tensor-product grid with the given node coordinates."""
    xs, ys, zs = (np.asarray(a, dtype=float) for a in (xs, ys, zs))
    nx, ny, nz = len(xs) - 1, len(ys) - 1, len(zs) - 1
    pts = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)  # [i, j, k, :]
    X = pts.transpose(2, 1, 0, 3).reshape(-1, 3)  # node id = i + (nx+1) * (j + (ny+1) * k)
    sx, sy, sz = 1, nx + 1, (nx + 1) * (ny + 1)
    i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    base = (i * sx + j * sy + k * sz).ravel()
    corner = np.array([ci * sx + cj * sy + ck * sz for ck in (0, 1) for cj in (0, 1) for ci in (0, 1)])
    T = base[:, None, None] + corner[_KUHN][None]
    return TetMesh(X, T.reshape(-1, 4))


def box(size=(1.0, 1.0, 1.0), res=(4, 4, 4), origin=(0.0, 0.0, 0.0)) -> TetMesh:
    """Axis-aligned box split into ``6 * nx * ny * nz`` tetrahedra."""
    axes = [np.linspace(0.0, s, int(r) + 1) + o for s, r, o in zip(size, res, origin)]
    return grid(*axes)


def _cube_to_sphere(p: np.ndarray) -> np.ndarray:
    """Smooth map from the cube [-1, 1]^3 onto the unit ball ("spherified cube")."""
    x, y, z = p.T
    x2, y2, z2 = x * x, y * y, z * z
    return np.stack(
        [
            x * np.sqrt(1 - y2 / 2 - z2 / 2 + y2 * z2 / 3),
            y * np.sqrt(1 - z2 / 2 - x2 / 2 + z2 * x2 / 3),
            z * np.sqrt(1 - x2 / 2 - y2 / 2 + x2 * y2 / 3),
        ],
        axis=1,
    )


def ball(radius=1.0, res=8, center=(0.0, 0.0, 0.0)) -> TetMesh:
    """Solid ball meshed by mapping a ``res^3`` cube onto the sphere."""
    m = box((2.0, 2.0, 2.0), (res, res, res), origin=(-1.0, -1.0, -1.0))
    return TetMesh(_cube_to_sphere(m.X) * radius + np.asarray(center, float), m.T)


def ellipsoid(semi_axes=(1.0, 1.0, 1.0), res=8, center=(0.0, 0.0, 0.0)) -> TetMesh:
    m = ball(1.0, res)
    return TetMesh(m.X * np.asarray(semi_axes, float) + np.asarray(center, float), m.T)


def ball_octant(radius=1.0, res=8, grading=1.0) -> TetMesh:
    """The octant x >= 0, y >= 0, z <= 0 of :func:`ball` (for symmetric models).

    ``res`` cells span each octant edge, so ``grading=1`` gives the element size
    of ``ball(radius, 2 * res)``. ``grading > 1`` clusters elements towards the
    south pole ``(0, 0, -radius)``, where a contact patch forms.
    """
    u = np.linspace(0.0, 1.0, int(res) + 1) ** grading
    m = grid(u, u, u - 1.0)
    return TetMesh(_cube_to_sphere(m.X) * radius, m.T)


def merge(meshes) -> TetMesh:
    """Concatenate several meshes into one (bodies stay disconnected)."""
    Xs, Ts, offset = [], [], 0
    for m in meshes:
        Xs.append(m.X)
        Ts.append(m.T + offset)
        offset += m.n_nodes
    return TetMesh(np.concatenate(Xs), np.concatenate(Ts))
