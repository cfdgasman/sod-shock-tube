import numpy as np
import pytest

from euler1d import State, sample, solve, star_state
from euler1d.fv import hllc, prim_to_cons

SOD_L, SOD_R = State(1.0, 0.0, 1.0), State(0.125, 0.0, 0.1)


def test_sod_star_state_matches_toro():
    # Toro (2009), Table 4.2, test 1
    p, u = star_state(SOD_L, SOD_R)
    assert p == pytest.approx(0.30313, abs=1e-5)
    assert u == pytest.approx(0.92745, abs=1e-5)


def test_toro_test3_star_state():
    p, u = star_state(State(1.0, 0.0, 1000.0), State(1.0, 0.0, 0.01))
    assert p == pytest.approx(460.894, abs=1e-3)
    assert u == pytest.approx(19.5975, abs=1e-4)


def test_hllc_is_consistent():
    w = (np.array([0.7]), np.array([0.3]), np.array([2.0]))
    f = hllc(w, w, 1.4)[:, 0]
    rho, u, p = 0.7, 0.3, 2.0
    e = prim_to_cons(rho, u, p, 1.4)[2]
    assert np.allclose(f, [rho * u, rho * u * u + p, u * (e + p)])


def _sod(n, **kw):
    xc = (np.arange(n) + 0.5) / n
    rho0 = np.where(xc < 0.5, 1.0, 0.125)
    p0 = np.where(xc < 0.5, 1.0, 0.1)
    return solve(rho0, np.zeros(n), p0, 0.2, **kw)


def test_conservation():
    # before waves reach the boundaries, total mass is exactly conserved
    x, rho, _, _ = _sod(200)
    assert rho.mean() == pytest.approx(0.5625, rel=1e-12)


@pytest.mark.parametrize("order,tol", [(1, 0.015), (2, 0.004)])
def test_sod_l1_error(order, tol):
    x, rho, _, _ = _sod(200, order=order)
    rho_e, _, _ = sample(x, 0.2, SOD_L, SOD_R)
    assert np.abs(rho - rho_e).mean() < tol


def test_muscl_beats_first_order():
    errs = []
    for order in (1, 2):
        x, rho, _, _ = _sod(200, order=order)
        errs.append(np.abs(rho - sample(x, 0.2, SOD_L, SOD_R)[0]).mean())
    assert errs[1] < 0.5 * errs[0]
