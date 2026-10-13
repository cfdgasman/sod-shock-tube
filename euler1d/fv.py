"""Finite-volume solver for the 1D Euler equations.

* Conserved variables U = (rho, rho u, E), ideal gas p = (gamma - 1)(E - rho u^2 / 2)
* HLLC approximate Riemann solver (Toro, Chapter 10), Davis wave-speed estimates
* Reconstruction: first order (piecewise constant) or MUSCL with a slope limiter
  applied to the primitive variables
* Time integration: SSP-RK2 (Heun), time step from a CFL condition
* Transmissive (zero-gradient) boundaries through two ghost cells
"""

from __future__ import annotations

import numpy as np

NG = 2  # ghost cells per side


def prim_to_cons(rho, u, p, gamma):
    return np.array([rho, rho * u, p / (gamma - 1) + 0.5 * rho * u * u])


def cons_to_prim(q, gamma):
    rho = q[0]
    u = q[1] / rho
    p = (gamma - 1) * (q[2] - 0.5 * rho * u * u)
    return rho, u, p


def _flux(rho, u, p, gamma):
    e = p / (gamma - 1) + 0.5 * rho * u * u
    return np.array([rho * u, rho * u * u + p, u * (e + p)])


def hllc(wl, wr, gamma):
    """HLLC flux between left/right primitive states (arrays of rho, u, p)."""
    rl, ul, pl = wl
    rr, ur, pr = wr
    cl = np.sqrt(gamma * pl / rl)
    cr = np.sqrt(gamma * pr / rr)
    sl = np.minimum(ul - cl, ur - cr)
    sr = np.maximum(ul + cl, ur + cr)
    ss = (pr - pl + rl * ul * (sl - ul) - rr * ur * (sr - ur)) / (rl * (sl - ul) - rr * (sr - ur))

    ql = prim_to_cons(rl, ul, pl, gamma)
    qr = prim_to_cons(rr, ur, pr, gamma)
    fl = _flux(rl, ul, pl, gamma)
    fr = _flux(rr, ur, pr, gamma)

    def star(q, r, u, p, s):
        fac = r * (s - u) / (s - ss)
        return fac * np.array([np.ones_like(r), ss, q[2] / r + (ss - u) * (ss + p / (r * (s - u)))])

    fsl = fl + sl * (star(ql, rl, ul, pl, sl) - ql)
    fsr = fr + sr * (star(qr, rr, ur, pr, sr) - qr)
    return np.where(sl >= 0, fl, np.where(ss >= 0, fsl, np.where(sr > 0, fsr, fr)))


LIMITERS = {
    "minmod": lambda a, b: np.where(a * b > 0, np.sign(a) * np.minimum(abs(a), abs(b)), 0.0),
    "vanleer": lambda a, b: np.where(a * b > 0, 2 * a * b / (a + b + 1e-300), 0.0),
    "mc": lambda a, b: np.where(
        a * b > 0,
        np.sign(a) * np.minimum.reduce([2 * abs(a), 2 * abs(b), 0.5 * abs(a + b)]),
        0.0,
    ),
}


def _rhs(q, dx, gamma, order, limiter):
    q = q.copy()
    q[:, :NG] = q[:, NG : NG + 1]  # transmissive boundaries
    q[:, -NG:] = q[:, -NG - 1 : -NG]
    w = np.array(cons_to_prim(q, gamma))
    if order == 1:
        wl, wr = w[:, NG - 1 : -NG], w[:, NG : -NG + 1 or None]
    else:
        slope = LIMITERS[limiter](w[:, 1:-1] - w[:, :-2], w[:, 2:] - w[:, 1:-1])  # cells 1..n-2
        left_face = w[:, 1:-1] + 0.5 * slope  # value at right face of each cell
        right_face = w[:, 1:-1] - 0.5 * slope  # value at left face of each cell
        # interface between cell i and i+1 for i = NG-1 .. last interior
        wl = left_face[:, NG - 2 : -NG + 1]
        wr = right_face[:, NG - 1 :]
    f = hllc(wl, wr, gamma)
    dq = np.zeros_like(q)
    dq[:, NG:-NG] = -(f[:, 1:] - f[:, :-1]) / dx
    return dq


def solve(rho0, u0, p0, t_end, x0=0.0, x1=1.0, cfl=0.5, gamma=1.4, order=2, limiter="vanleer"):
    """Advance cell-averaged initial data to t_end. Returns (x_centres, rho, u, p)."""
    n = len(rho0)
    dx = (x1 - x0) / n
    x = x0 + (np.arange(n) + 0.5) * dx
    q = np.zeros((3, n + 2 * NG))
    q[:, NG:-NG] = prim_to_cons(np.asarray(rho0, float), np.asarray(u0, float), np.asarray(p0, float), gamma)

    t = 0.0
    while t < t_end:
        rho, u, p = cons_to_prim(q[:, NG:-NG], gamma)
        smax = np.max(np.abs(u) + np.sqrt(gamma * p / rho))
        dt = min(cfl * dx / smax, t_end - t)
        q1 = q + dt * _rhs(q, dx, gamma, order, limiter)
        q = 0.5 * (q + q1 + dt * _rhs(q1, dx, gamma, order, limiter))
        t += dt
    rho, u, p = cons_to_prim(q[:, NG:-NG], gamma)
    return x, rho, u, p
