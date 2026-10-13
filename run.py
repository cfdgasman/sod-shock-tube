"""Run the shock-tube test cases, compare with the exact solution and write figures to docs/."""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from euler1d import State, sample, solve

GAMMA = 1.4

# name: (left, right, t_end, x0)
CASES = {
    "sod": (State(1.0, 0.0, 1.0), State(0.125, 0.0, 0.1), 0.2, 0.5),
    "lax": (State(0.445, 0.698, 3.528), State(0.5, 0.0, 0.571), 0.14, 0.5),
    "toro3": (State(1.0, 0.0, 1000.0), State(1.0, 0.0, 0.01), 0.012, 0.5),
}
TITLES = {
    "sod": "Sod shock tube, t = 0.2",
    "lax": "Lax shock tube, t = 0.14",
    "toro3": "Toro test 3 (strong blast, p ratio 10⁵), t = 0.012",
}


def run_case(name, n, order=2, limiter="vanleer"):
    left, right, t_end, x0 = CASES[name]
    xc = (np.arange(n) + 0.5) / n
    rho0 = np.where(xc < x0, left.rho, right.rho)
    u0 = np.where(xc < x0, left.u, right.u)
    p0 = np.where(xc < x0, left.p, right.p)
    return solve(rho0, u0, p0, t_end, gamma=GAMMA, order=order, limiter=limiter)


def exact(name, x):
    left, right, t_end, x0 = CASES[name]
    return sample(x, t_end, left, right, x0=x0, gamma=GAMMA)


def l1_density_error(name, n, **kw):
    x, rho, _, _ = run_case(name, n, **kw)
    # compare with the exact cell average (sub-sampled) rather than the point value
    sub = 16
    xs = (np.arange(n * sub) + 0.5) / (n * sub)
    rho_e = exact(name, xs)[0].reshape(n, sub).mean(axis=1)
    return np.abs(rho - rho_e).mean()


def plot_case(name, n=200):
    xf = np.linspace(0, 1, 2000)
    re_, ue, pe = exact(name, xf)
    ee = pe / ((GAMMA - 1) * re_)
    x1, r1, u1, p1 = run_case(name, n, order=1)
    x2, r2, u2, p2 = run_case(name, n, order=2)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    panels = [
        ("Density ρ", re_, r1, r2),
        ("Velocity u", ue, u1, u2),
        ("Pressure p", pe, p1, p2),
        ("Internal energy e", ee, p1 / ((GAMMA - 1) * r1), p2 / ((GAMMA - 1) * r2)),
    ]
    for ax, (label, ex, a, b) in zip(axes.flat, panels):
        ax.plot(xf, ex, "k-", lw=1.4, label="Exact")
        ax.plot(x1, a, ".", ms=3.5, color="#999999", label="1st order (Godunov + HLLC)")
        ax.plot(x2, b, ".", ms=3.5, color="#d62728", label="MUSCL + HLLC + SSP-RK2")
        ax.set_title(label)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("x")
    fig.suptitle(f"{TITLES[name]}, {n} cells")
    fig.tight_layout()
    fig.savefig(f"docs/{name}.png", dpi=120)
    plt.close(fig)


def main():
    t0 = time.perf_counter()
    for name in CASES:
        plot_case(name)
    print(f"figures written in {time.perf_counter() - t0:.1f} s")

    ns = [100, 200, 400, 800, 1600]
    schemes = [
        ("1st order", dict(order=1)),
        ("MUSCL minmod", dict(order=2, limiter="minmod")),
        ("MUSCL van Leer", dict(order=2, limiter="vanleer")),
        ("MUSCL MC", dict(order=2, limiter="mc")),
    ]
    errs = {label: [l1_density_error("sod", n, **kw) for n in ns] for label, kw in schemes}

    print("\nSod, L1 density error vs exact cell averages")
    print("| Cells | " + " | ".join(errs) + " |")
    print("|---" * (len(errs) + 1) + "|")
    for k, n in enumerate(ns):
        print(f"| {n} | " + " | ".join(f"{errs[s][k]:.2e}" for s in errs) + " |")
    print("| **order** | " + " | ".join(
        f"**{np.polyfit(np.log(ns), np.log(errs[s]), 1)[0] * -1:.2f}**" for s in errs) + " |")

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for s in errs:
        ax.loglog(ns, errs[s], "o-", label=s)
    ax.loglog(ns, errs["1st order"][0] * (ns[0] / np.array(ns)) ** 0.5, "k:", lw=1, label="slope ½")
    ax.loglog(ns, errs["MUSCL van Leer"][0] * (ns[0] / np.array(ns)), "k--", lw=1, label="slope 1")
    ax.set(xlabel="cells", ylabel="L1 error in ρ", title="Sod: grid convergence")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("docs/convergence.png", dpi=120)


if __name__ == "__main__":
    main()
