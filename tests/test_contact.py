import numpy as np
import pytest

import softfem as sf
from softfem.contact import ContactSet, Friction, friction_f0, friction_f1_over_y


def test_barrier_derivatives():
    dhat = 1e-2
    d = np.linspace(1e-4, 1.5 * dhat, 200)
    b, b1, b2 = sf.barrier(d, dhat)
    h = 1e-8
    bp, _, _ = sf.barrier(d + h, dhat)
    bm, _, _ = sf.barrier(d - h, dhat)
    np.testing.assert_allclose(b1, (bp - bm) / (2 * h), rtol=1e-5, atol=1e-12)
    _, b1p, _ = sf.barrier(d + h, dhat)
    _, b1m, _ = sf.barrier(d - h, dhat)
    np.testing.assert_allclose(b2, (b1p - b1m) / (2 * h), rtol=1e-5, atol=1e-9)
    inside = d < dhat
    assert np.all(b[~inside] == 0) and np.all(b[inside] > 0)
    assert np.all(b1[inside] < 0) and np.all(b2[inside] > 0)  # repulsive and convex


def test_friction_smoothing_is_c1():
    eps = 1e-3
    y = np.linspace(0, 3 * eps, 301)
    h = 1e-9
    f1 = friction_f1_over_y(y, eps) * y
    num = (friction_f0(y + h, eps) - friction_f0(np.maximum(y - h, 0), eps)) / (y + h - np.maximum(y - h, 0))
    np.testing.assert_allclose(f1[1:], num[1:], rtol=1e-4, atol=1e-6)
    # continuity of value and slope at y = eps
    np.testing.assert_allclose(friction_f0(np.array([eps]), eps), [eps])
    np.testing.assert_allclose(friction_f1_over_y(np.array([eps]), eps) * eps, [1.0])


def _contact_fixture(obstacle):
    """40 nodes at distances in (0.0015, 0.0095) from the obstacle (dhat = 0.01)."""
    rng = np.random.default_rng(0)
    d = rng.uniform(0.0015, 0.0095, 40)
    if isinstance(obstacle, sf.Plane):
        n = obstacle.normal_vec
        t = rng.uniform(-1, 1, (40, 3))
        t -= (t @ n)[:, None] * n
        x = obstacle.point + t + d[:, None] * n
    else:
        u = rng.standard_normal((40, 3)) * [0.1, 0.1, 0.0] + [0, 0, 1.0]
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        x = obstacle.center + (obstacle.radius + d)[:, None] * u
    return x, ContactSet([obstacle], np.arange(40), rng.uniform(0.5, 1.5, 40), dhat=0.01, kappa=10.0)


@pytest.mark.parametrize(
    "obstacle",
    [sf.Plane(normal=(0.1, 0.0, 1.0)), sf.Sphere(center=(0.0, 0.0, -10.0), radius=10.0)],
    ids=["plane", "sphere"],
)
def test_contact_gradient(obstacle):
    x, cs = _contact_fixture(obstacle)
    assert np.isfinite(cs.energy(x))
    g = cs.gradient(x)
    v = np.random.default_rng(1).standard_normal(x.shape)
    h = 1e-8
    num = (cs.energy(x + h * v) - cs.energy(x - h * v)) / (2 * h)
    np.testing.assert_allclose(np.sum(g * v), num, rtol=1e-5)


def test_plane_contact_hessian():
    x, cs = _contact_fixture(sf.Plane())
    H = cs.hessian_blocks(x, len(x))
    v = np.random.default_rng(2).standard_normal(x.shape)
    h = 1e-8
    num = (cs.gradient(x + h * v) - cs.gradient(x - h * v)) / (2 * h)
    np.testing.assert_allclose(np.einsum("kij,kj->ki", H, v), num, rtol=1e-5, atol=1e-8)


def test_ccd_step_never_penetrates():
    rng = np.random.default_rng(3)
    for obstacle in (sf.Plane(), sf.Sphere(radius=0.5)):
        for _ in range(50):
            x = rng.standard_normal((30, 3))
            x = x[obstacle.distance(x) > 1e-3]
            p = 3.0 * rng.standard_normal(x.shape)
            alpha = min(1.0, obstacle.max_step(x, p))
            assert np.all(obstacle.distance(x + alpha * p) > 0)


def test_friction_derivatives():
    plane = sf.Plane(mu=0.7)
    x0, cs = _contact_fixture(plane)
    fr = Friction()
    fr.update(cs, x0, eps=1e-3)
    rng = np.random.default_rng(4)
    # displacements both inside and outside the static (smoothed) zone
    x = x0 + rng.standard_normal(x0.shape) * [2e-3, 2e-3, 1e-4]
    g = fr.gradient(x)
    H = fr.hessian_blocks(x, len(x))
    v = rng.standard_normal(x.shape)
    h = 1e-9
    num = (fr.energy(x + h * v) - fr.energy(x - h * v)) / (2 * h)
    np.testing.assert_allclose(np.sum(g * v), num, rtol=1e-5)
    numg = (fr.gradient(x + h * v) - fr.gradient(x - h * v)) / (2 * h)
    np.testing.assert_allclose(np.einsum("kij,kj->ki", H, v), numg, rtol=1e-4, atol=1e-6 * np.abs(numg).max())
    assert np.linalg.eigvalsh(H).min() > -1e-9
