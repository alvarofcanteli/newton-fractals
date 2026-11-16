"""E2: cost (iterations and evaluations) of the five methods until convergence.

z^3 - 1 is the original experiment (with its histograms). The cost table is also computed, without
figures, for z^3 - 2z + 2, z^6 - 1 and the random degree-5 polynomial (seed 2026) with the same grid
(window [-2, 2]^2, N = 1000), tolerance, max_iter and delta.

Generates results/e2_cost.csv (one block of rows per polynomial), results/e2_summary.csv (adds the
percentages of converged, non-converged and failed points) and figures/e2_hist.png (z^3 - 1 only).
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
from nfractals.plotting import poly_latex  # noqa: E402
from nfractals.polynomials import POLYNOMIALS, make_polynomial, unique_roots  # noqa: E402
from nfractals.runner import (  # noqa: E402
    CONVERGED,
    FAILED,
    NOT_CONVERGED,
    run_basins,
    single_point,
)

POLYS = ["z^3 - 1", "z^3 - 2z + 2", "z^6 - 1", "random degree 5"]
N = 1000
TOL = 1e-10
H_DAMPED = 0.7
DELTA = 1e-3

x = np.linspace(-2, 2, N)
z0 = x[None, :] + 1j * x[:, None]

columns = [
    "polynomial",
    "method",
    "max_iter",
    "evals per step",
    "initial evals",
    "mean iter",
    "median iter",
    "mean evals",
    "median evals",
]
summary_columns = [
    "polynomial", "method", "max_iter", "converged (%)", "not converged (%)", "failed (%)",
    "mean iter", "median iter", "mean evals", "median evals",
]

rows = []
summary_rows = []
iterations = {}  # polynomial -> iterations of the converged points of each method
hist_methods = None
for poly_name in POLYS:
    f, df, d2f, roots_np = make_polynomial(POLYNOMIALS[poly_name])
    roots = unique_roots(roots_np)

    # (name, step, max_iter, evals per step, initial evals, initial state)
    # evals per point = iterations * evals per step + initial evals.
    methods = [
        ("Newton", single_point(lambda z: newton_step(z, f, df)), 50, 2, 0, None),
        ("Halley", single_point(lambda z: halley_step(z, f, df, d2f)), 50, 3, 0, None),
        (f"Damped Newton (h={H_DAMPED})",
         single_point(lambda z: newton_step(z, f, df, h=H_DAMPED)), 200, 2, 0, None),
        ("Secant", lambda s: secant_step(s, f), 100, 1, 2, secant_init(z0, DELTA)),
        ("Müller", lambda s: muller_step(s, f), 100, 1, 3, muller_init(z0, DELTA)),
    ]
    if poly_name == "z^3 - 1":
        hist_methods = methods
    iterations[poly_name] = []
    for name, step, max_iter, cost, ini, init in methods:
        _, iters, status = run_basins(z0, step, roots, TOL, max_iter, init=init)
        it = iters[status == CONVERGED]
        iterations[poly_name].append(it)
        rows.append([
            poly_name,
            name,
            max_iter,
            cost,
            ini,
            round(float(it.mean()), 3),
            round(float(np.median(it)), 3),
            round(float(it.mean() * cost + ini), 3),
            round(float(np.median(it) * cost + ini), 3),
        ])
        summary_rows.append([
            poly_name, name, max_iter,
            round(100 * float(np.mean(status == CONVERGED)), 4),
            round(100 * float(np.mean(status == NOT_CONVERGED)), 4),
            round(100 * float(np.mean(status == FAILED)), 4),
            round(float(it.mean()), 3), round(float(np.median(it)), 3),
            round(float(it.mean() * cost + ini), 3), round(float(np.median(it) * cost + ini), 3),
        ])

# On-screen table.
texts = [[str(c) if not isinstance(c, float) else f"{c:.3f}" for c in row] for row in rows]
widths = [max(len(columns[i]), *(len(t[i]) for t in texts)) for i in range(len(columns))]
align = lambda row: "  ".join(
    c.ljust(widths[i]) if i <= 1 else c.rjust(widths[i]) for i, c in enumerate(row)
)
print(align(columns))
print("  ".join("-" * w for w in widths))
for t in texts:
    print(align(t))
print("\nPercentage of the grid that does not converge / fails:")
for r in summary_rows:
    print(f"  {r[0]} | {r[1]}: not converged {r[4]}%, failed {r[5]}%")

# CSV.
output_csv = PROJECT_ROOT / "results" / "e2_cost.csv"
with open(output_csv, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(columns)
    w.writerows(rows)
print(f"\nTable saved to {output_csv}")
output_summary = PROJECT_ROOT / "results" / "e2_summary.csv"
with open(output_summary, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(summary_columns)
    w.writerows(summary_rows)
print(f"Summary saved to {output_summary}")

# Histograms (z^3 - 1 only) with the same x-axis scale.
iterations = iterations["z^3 - 1"]
methods = hist_methods
it_max = max(int(it.max()) for it in iterations)
bins = np.arange(1, it_max + 2) - 0.5

fig, axes = plt.subplots(
    len(methods), 1, figsize=(8, 3 * len(methods)), sharex=True, constrained_layout=True
)
for ax, (name, _, max_iter, *_), it in zip(axes, methods, iterations):
    ax.hist(it, bins=bins)
    ax.set_title(f"{name}, max_iter={max_iter}")
    ax.set_ylabel("points")
axes[-1].set_xlabel("iterations to converge")
axes[-1].set_xlim(0, it_max + 1)
fig.suptitle(f"Iterations to converge, {poly_latex('z^3 - 1')} on [−2, 2]², N = {N} (tol={TOL:g})")

output_fig = PROJECT_ROOT / "figures" / "e2_hist.png"
fig.savefig(output_fig, dpi=100, bbox_inches="tight")
print(f"Figure saved to {output_fig}")
