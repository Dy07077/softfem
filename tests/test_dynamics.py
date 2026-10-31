import numpy as np
import pytest

import softfem as sf
from softfem.validation import cantilever_tip_deflection, hertz_contact, incline_acceleration


def test_free_fall_matches_backward_euler_exactly():
    h, N, g = 0.01, 25, 9.81
    sim = sf.Simulator(sf.ball(0.1, 3), sf.StableNeoHookean(1e5, 0.3), dt=h, gravity=(0, 0, -g))
    z0 = sim.center_of_mass()[2]
    for _ in range(N):
        sim.step()
    # backward Euler: v_N = -g N h,  z_N = z_0 - g h^2 N (N + 1) / 2
    assert sim.center_of_mass()[2] == pytest.approx(z0 - g * h * h * N * (N + 1) / 2, abs=1e-10)
    np.testing.assert_allclose(sim.v[:, 2], -g * N * h, atol=1e-9)


def test_linear_momentum_is_conserved():
    mesh = sf.box((0.2, 0.1, 0.1), (4, 2, 2))
    sim = sf.Simulator(mesh, sf.StableNeoHookean(2e4, 0.4), dt=0.01, gravity=(0, 0, 0))
    rng = np.random.default_rng(0)
    c = mesh.X.mean(0)
    sim.v = np.array([0.3, -0.2, 0.1]) + np.cross([0.0, 0.0, 6.0], mesh.X - c) + 0.2 * rng.standard_normal(mesh.X.shape)
    p0 = (sim.mass[:, None] * sim.v).sum(0)
    for _ in range(30):
        sim.step()
    p = (sim.mass[:, None] * sim.v).sum(0)
    np.testing.assert_allclose(p, p0, atol=1e-8)


def test_body_settles_on_ground_without_penetration():
    mesh = sf.box((0.1, 0.1, 0.1), (3, 3, 3), origin=(-0.05, -0.05, 0.02))
    sim = sf.Simulator(mesh, sf.StableNeoHookean(5e4, 0.4), dt=0.01, obstacles=[sf.Plane(mu=0.5)])
    gaps = []
    for _ in range(150):
        sim.step()
        gaps.append(sim.contact.min_distance(sim.x))
    assert min(gaps) > 0.0
    weight = sim.mass.sum() * 9.81
    assert sim.contact_forces()[0][2] == pytest.approx(weight, rel=1e-2)
    assert sim.kinetic_energy() < 1e-6 * weight * 0.1


def test_sphere_obstacle_is_never_penetrated():
    mesh = sf.box((0.3, 0.3, 0.03), (8, 8, 1), origin=(-0.15, -0.15, 0.13))
    sim = sf.Simulator(mesh, sf.StableNeoHookean(2e4, 0.3), dt=0.01, obstacles=[sf.Sphere((0, 0, 0), 0.12, mu=0.2)])
    for _ in range(40):
        sim.step()
        assert sim.contact.min_distance(sim.x) > 0.0
    assert sim.contact_forces()[0][2] > 0.0  # the sheet is resting on the ball


@pytest.mark.parametrize("theta, sticks", [(10.0, True), (30.0, False)])
def test_coulomb_friction_on_incline(theta, sticks):
    a_fem, a_theory, slip = incline_acceleration(theta, mu=0.3, duration=0.6)
    if sticks:
        assert a_theory == 0.0 and abs(a_fem) < 1e-3 and slip < 1e-3
    else:
        assert a_fem == pytest.approx(a_theory, rel=2e-3)


def test_hertz_contact_power_law():
    r = hertz_contact(res=6, grading=1.6, n_steps=4)
    slope = np.polyfit(np.log(r.delta[1:]), np.log(r.force[1:]), 1)[0]
    assert 1.4 < slope < 1.6  # Hertz: F ~ delta^(3/2)
    ratio = r.force[-1] / r.force_hertz[-1]
    assert 1.0 < ratio < 1.4  # linear tets are stiff on coarse meshes; converges with refinement
    # the measured radius is quantised by the node spacing on this coarse mesh
    np.testing.assert_allclose(r.radius[-1], r.radius_hertz[-1], rtol=0.25)


def test_cantilever_converges_to_beam_theory():
    ratios = [cantilever_tip_deflection(n)[0] / cantilever_tip_deflection(n)[1] for n in (1, 2, 4)]
    assert ratios[0] < ratios[1] < ratios[2] < 1.0
    assert ratios[2] > 0.8
