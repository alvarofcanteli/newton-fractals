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
from nfractals.polynomials import POLYNOMIALS, make_polynomial  # noqa: E402
from nfractals.runner import CONVERGED, run_basins, single_point  # noqa: E402

N = 1000
TOL = 1e-10
H_DAMPED = 0.7
DELTA = 1e-3

f, df, d2f, roots = make_polynomial(POLYNOMIALS["z^3 - 1"])

x = np.linspace(-2, 2, N)
z0 = x[None, :] + 1j * x[:, None]

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

columns = [
    "method",
    "max_iter",
    "evals per step",
    "initial evals",
    "mean iter",
    "median iter",
    "mean evals",
    "median evals",
]

rows = []
iterations = []
for name, step, max_iter, cost, ini, init in methods:
    _, iters, status = run_basins(z0, step, f, roots, TOL, max_iter, init=init)
    it = iters[status == CONVERGED]
    iterations.append(it)
    rows.append([
        name,
        max_iter,
        cost,
        ini,
        round(float(it.mean()), 3),
        round(float(np.median(it)), 3),
        round(float(it.mean() * cost + ini), 3),
        round(float(np.median(it) * cost + ini), 3),
    ])

# On-screen table.
texts = [[str(c) if not isinstance(c, float) else f"{c:.3f}" for c in row] for row in rows]
widths = [max(len(columns[i]), *(len(t[i]) for t in texts)) for i in range(len(columns))]
align = lambda row: "  ".join(
    c.ljust(widths[i]) if i == 0 else c.rjust(widths[i]) for i, c in enumerate(row)
)
print(align(columns))
print("  ".join("-" * w for w in widths))
for t in texts:
    print(align(t))

# CSV.
output_csv = PROJECT_ROOT / "results" / "e2_cost.csv"
with open(output_csv, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(columns)
    w.writerows(rows)
print(f"\nTable saved to {output_csv}")

# Histograms with the same x-axis scale.
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

output_fig = PROJECT_ROOT / "figures" / "e2_hist.png"
fig.savefig(output_fig, dpi=100, bbox_inches="tight")
print(f"Figure saved to {output_fig}")
