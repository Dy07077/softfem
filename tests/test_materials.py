import numpy as np
import pytest

import softfem as sf
from softfem.materials import project_psd

MATERIALS = [
    sf.LinearElastic(1e5, 0.3),
    sf.StableNeoHookean(1e5, 0.3),
    sf.StableNeoHookean(2e4, 0.45),
]
IDS = ["linear", "snh-0.3", "snh-0.45"]


def random_F(seed, n=6, inverted=False):
    rng = np.random.default_rng(seed)
    F = np.eye(3) + 0.3 * rng.standard_normal((n, 3, 3))
    if inverted:
        F[:, :, 0] *= -1.0
    return F


@pytest.mark.parametrize("mat", MATERIALS, ids=IDS)
def test_rest_state_is_stress_free(mat):
    F = np.eye(3)[None]
    assert abs(mat.energy_density(F)[0]) < 1e-12 * mat.E
    assert np.abs(mat.pk1(F)).max() < 1e-10 * mat.E


@pytest.mark.parametrize("mat", MATERIALS, ids=IDS)
@pytest.mark.parametrize("inverted", [False, True])
def test_pk1_matches_finite_differences(mat, inverted):
    F = random_F(1, inverted=inverted)
    P = mat.pk1(F)
    h = 1e-6
    for i in range(3):
        for j in range(3):
            dF = np.zeros((3, 3))
            dF[i, j] = h
            num = (mat.energy_density(F + dF) - mat.energy_density(F - dF)) / (2 * h)
            np.testing.assert_allclose(P[:, i, j], num, rtol=1e-6, atol=1e-6 * mat.E)


@pytest.mark.parametrize("mat", MATERIALS, ids=IDS)
@pytest.mark.parametrize("inverted", [False, True])
def test_hessian_matches_finite_differences(mat, inverted):
    F = random_F(2, inverted=inverted)
    H = mat.hessian(F, project=False)
    h = 1e-6
    for k in range(3):
        for l in range(3):
            dF = np.zeros((3, 3))
            dF[k, l] = h
            num = (mat.pk1(F + dF) - mat.pk1(F - dF)).reshape(len(F), 9) / (2 * h)
            np.testing.assert_allclose(H[:, :, 3 * k + l], num, rtol=1e-6, atol=1e-6 * mat.E)


def test_neo_hookean_linearises_to_linear_elasticity():
    E, nu = 3e5, 0.35
    I = np.eye(3)[None]
    H_snh = sf.StableNeoHookean(E, nu).hessian(I, project=False)[0]
    H_lin = sf.LinearElastic(E, nu).hessian(I)[0]
    np.testing.assert_allclose(H_snh, H_lin, atol=1e-9 * E)


def test_neo_hookean_is_rotation_invariant():
    from scipy.spatial.transform import Rotation

    mat = sf.StableNeoHookean(1e5, 0.4)
    F = random_F(3, n=20)
    R = Rotation.random(20, random_state=4).as_matrix()
    np.testing.assert_allclose(mat.energy_density(R @ F), mat.energy_density(F), rtol=1e-10, atol=1e-6)


def test_psd_projection():
    F = random_F(5, n=50, inverted=True)
    H = sf.StableNeoHookean(1e5, 0.45).hessian(F, project=False)
    assert np.linalg.eigvalsh(H).min() < 0  # the raw Hessian is indefinite here
    Hp = project_psd(H)
    assert np.linalg.eigvalsh(Hp).min() > -1e-8 * np.abs(H).max()
    # projection leaves positive-definite matrices untouched
    spd = np.eye(9)[None] * 3.0
    np.testing.assert_allclose(project_psd(spd), spd)


def test_lame_rejects_invalid_poisson_ratio():
    with pytest.raises(ValueError):
        sf.lame(1e5, 0.5)


def test_per_element_parameters_match_homogeneous_materials():
    F = random_F(6, n=4)
    E = np.array([1e4, 1e4, 5e5, 5e5])
    het = sf.StableNeoHookean(E, 0.4)
    soft, stiff = sf.StableNeoHookean(1e4, 0.4), sf.StableNeoHookean(5e5, 0.4)
    for fn in ("energy_density", "pk1"):
        got = getattr(het, fn)(F)
        np.testing.assert_allclose(got[:2], getattr(soft, fn)(F[:2]))
        np.testing.assert_allclose(got[2:], getattr(stiff, fn)(F[2:]))
    H = het.hessian(F, project=False)
    np.testing.assert_allclose(H[:2], soft.hessian(F[:2], project=False))
    np.testing.assert_allclose(H[2:], stiff.hessian(F[2:], project=False))
