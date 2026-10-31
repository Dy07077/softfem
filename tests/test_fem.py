import numpy as np
import scipy.sparse as sp

import softfem as sf
from softfem.fem import HessianAssembler


def deformed(mesh, seed=0, amp=0.05):
    rng = np.random.default_rng(seed)
    return mesh.X + amp * rng.standard_normal(mesh.X.shape)


def test_box_mesh_is_closed_and_conforming():
    m = sf.box((2.0, 1.0, 0.5), (4, 3, 2))
    assert m.n_tets == 6 * 4 * 3 * 2
    assert np.all(m.volumes() > 0)
    np.testing.assert_allclose(m.volumes().sum(), 1.0)
    faces = m.boundary_faces()
    assert len(faces) == 2 * 2 * (4 * 3 + 3 * 2 + 4 * 2)
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert np.all(counts == 2)  # watertight
    V, E, F = len(np.unique(faces)), len(counts), len(faces)
    assert V - E + F == 2  # sphere topology


def test_boundary_faces_point_outward():
    m = sf.ball(1.0, 6)
    f = m.boundary_faces()
    a, b, c = (m.X[f[:, k]] for k in range(3))
    n = np.cross(b - a, c - a)
    assert np.all(np.einsum("ij,ij->i", n, (a + b + c) / 3) > 0)


def test_ball_volume_converges():
    errs = [abs(sf.ball(1.0, r).volumes().sum() - 4 / 3 * np.pi) for r in (4, 8, 16)]
    # faceting error of a linear mesh is second order in the element size
    assert errs[0] / errs[1] > 3.5 and errs[1] / errs[2] > 3.5
    assert errs[2] / (4 / 3 * np.pi) < 5e-3


def test_elastic_gradient_matches_finite_differences():
    mesh = sf.ball(0.1, 3)
    el = sf.Elasticity(mesh, sf.StableNeoHookean(1e5, 0.4))
    x = deformed(mesh, amp=0.01)
    g = el.gradient(x)
    rng = np.random.default_rng(1)
    for _ in range(5):
        v = rng.standard_normal(x.shape)
        h = 1e-7
        num = (el.energy(x + h * v) - el.energy(x - h * v)) / (2 * h)
        np.testing.assert_allclose(g @ v.ravel(), num, rtol=1e-5)


def test_elastic_hessian_matches_finite_differences():
    mesh = sf.ball(0.1, 3)
    el = sf.Elasticity(mesh, sf.StableNeoHookean(1e5, 0.4))
    x = deformed(mesh, amp=0.01)
    asm = HessianAssembler(mesh.n_nodes, mesh.T)
    H = asm.assemble(el.hessian_blocks(x, project=False))
    v = np.random.default_rng(2).standard_normal(x.shape)
    h = 1e-7
    num = (el.gradient(x + h * v) - el.gradient(x - h * v)) / (2 * h)
    np.testing.assert_allclose(H @ v.ravel(), num, rtol=1e-5, atol=1e-6 * np.abs(num).max())


def test_assembler_matches_dense_scatter():
    mesh = sf.box((1, 1, 1), (2, 1, 1))
    rng = np.random.default_rng(3)
    eb = rng.standard_normal((mesh.n_tets, 12, 12))
    nb = rng.standard_normal((mesh.n_nodes, 3, 3))
    dense = np.zeros((3 * mesh.n_nodes,) * 2)
    dofs = (3 * mesh.T[:, :, None] + np.arange(3)).reshape(-1, 12)
    for e in range(mesh.n_tets):
        dense[np.ix_(dofs[e], dofs[e])] += eb[e]
    for i in range(mesh.n_nodes):
        d = 3 * i + np.arange(3)
        dense[np.ix_(d, d)] += nb[i]
    H = HessianAssembler(mesh.n_nodes, mesh.T).assemble(eb, nb)
    assert sp.isspmatrix_csr(H) or isinstance(H, sp.csr_matrix)
    np.testing.assert_allclose(H.toarray(), dense, atol=1e-12)


def test_rigid_motion_costs_no_energy():
    from scipy.spatial.transform import Rotation

    mesh = sf.ball(0.2, 4)
    el = sf.Elasticity(mesh, sf.StableNeoHookean(1e5, 0.45))
    R = Rotation.from_rotvec([0.3, -1.2, 0.7]).as_matrix()
    x = mesh.X @ R.T + np.array([1.0, -2.0, 0.5])
    assert abs(el.energy(x)) < 1e-9
    assert np.abs(el.gradient(x)).max() < 1e-6


def test_patch_test_reproduces_affine_deformation():
    """Prescribing an affine map on the boundary must give it everywhere."""
    mesh = sf.box((1.0, 1.0, 1.0), (3, 3, 3))
    A = np.array([[1.1, 0.05, 0.0], [0.0, 0.95, 0.1], [0.02, 0.0, 1.05]])
    c = np.array([0.1, -0.2, 0.3])
    target = mesh.X @ A.T + c
    sim = sf.Simulator(mesh, sf.StableNeoHookean(1e5, 0.3), quasistatic=True, gravity=(0, 0, 0), newton_tol=1e-10)
    surf = mesh.surface_nodes()
    sim.fix(surf, target[surf])
    sim.step()
    assert sim.history[-1].converged
    np.testing.assert_allclose(sim.x, target, atol=1e-8)
    vm = sim.elastic.von_mises(sim.x)
    np.testing.assert_allclose(vm, vm.mean(), rtol=1e-6)
