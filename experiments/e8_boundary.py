"""E8: boundary of the basins for the five methods, at N in {250, 500, 1000, 2000}.

Boundary = converged pixels with a converged neighbour of another root (same function as E6,
`nfractals.metrics.boundary_fraction`). Same tol, max_iter and delta as E2 and E3.
Generates results/e8_boundary.csv, results/e8_boundary_slope.csv and figures/e8_boundary.png.
The slope of log(boundary pixels) against log(N) is a rough box-counting dimension of the boundary.
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from nfractals.methods import (  # noqa: E402
    halley_step,
    muller_init,
    muller_step,
    newton_step,
    secant_init,
    secant_step,
)
from nfractals.metrics import boundary_mask  # noqa: E402
from nfractals.plotting import poly_latex  # noqa: E402
from nfractals.polynomials import POLYNOMIALS, make_polynomial, unique_roots  # noqa: E402
from nfractals.runner import CONVERGED, run_basins, single_point  # noqa: E402

NS = (250, 500, 1000, 2000)
TOL = 1e-10
H_DAMPED = 0.7
DELTA = 1e-3
POLYS = ["z^3 - 1", "z^3 - 2z + 2"]

columns = ["polynomial", "method", "N", "max_iter", "boundary pixels", "boundary (%)",
           "boundary pixels / N", "converged (%)"]
rows = []
slope_rows = []
counts = {}  # (poly, method) -> list of fractions per N

for poly_name in POLYS:
    f, df, d2f, r = make_polynomial(POLYNOMIALS[poly_name])
    roots = unique_roots(r)
    print(f"\n### {poly_name}", flush=True)
    per_method = {}
    for n in NS:
        x = np.linspace(-2, 2, n)
        z0 = x[None, :] + 1j * x[:, None]
        # (name, step, max_iter, initial state; None -> (z0,))
        methods = [
            ("Newton", single_point(lambda z: newton_step(z, f, df)), 50, None),
            ("Halley", single_point(lambda z: halley_step(z, f, df, d2f)), 50, None),
            (f"Damped Newton (h={H_DAMPED})",
             single_point(lambda z: newton_step(z, f, df, h=H_DAMPED)), 200, None),
            ("Secant", lambda s: secant_step(s, f), 100, secant_init(z0, DELTA)),
            ("Müller", lambda s: muller_step(s, f), 100, muller_init(z0, DELTA)),
        ]
        for name, step, max_iter, init in methods:
            root, _, status = run_basins(z0, step, roots, TOL, max_iter, init=init)
            count = int(boundary_mask(root, status).sum())
            p_conv = 100 * np.mean(status == CONVERGED)
            rows.append([poly_name, name, n, max_iter, count, round(100 * count / n**2, 4),
                         round(count / n, 4), round(p_conv, 4)])
            per_method.setdefault(name, []).append((n, count, 100 * count / n**2))
            print(f"- {poly_name} | {name} | N={n}: boundary {count} px "
                  f"({100 * count / n**2:.4f}%), {count / n:.3f} px per N, converged {p_conv:.2f}%",
                  flush=True)
    for name, vals in per_method.items():
        ns = np.array([v[0] for v in vals], dtype=float)
        cs = np.array([v[1] for v in vals], dtype=float)
        slope, intercept = np.polyfit(np.log(ns), np.log(cs), 1)
        slope_rows.append([poly_name, name, round(float(slope), 4)])
        counts[(poly_name, name)] = [v[2] for v in vals]
        print(f"  slope log(boundary px) vs log(N), {poly_name} | {name}: {slope:.4f}", flush=True)

with open(PROJECT_ROOT / "results" / "e8_boundary.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(columns)
    w.writerows(rows)
with open(PROJECT_ROOT / "results" / "e8_boundary_slope.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["polynomial", "method", "slope of log(boundary pixels) vs log(N), N = 250..2000"])
    w.writerows(slope_rows)

fig, axes = plt.subplots(1, len(POLYS), figsize=(11, 4.5), constrained_layout=True, sharey=False)
for ax, poly_name in zip(axes, POLYS):
    for (p, name), fr in counts.items():
        if p == poly_name:
            ax.plot(NS, fr, marker="o", label=name)
    ax.set_xscale("log", base=2)
    ax.set_xticks(NS)
    ax.set_xticklabels([str(n) for n in NS])
    ax.set_xlabel("N")
    ax.set_ylabel("boundary fraction (%)")
    ax.set_title(f"{poly_latex(poly_name)} on [−2, 2]² (tol={TOL:g})", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
fig.savefig(PROJECT_ROOT / "figures" / "e8_boundary.png", dpi=100, bbox_inches="tight")
plt.close(fig)
print("\nTables saved to results/e8_boundary.csv and results/e8_boundary_slope.csv")
