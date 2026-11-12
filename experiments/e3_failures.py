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
from nfractals.plotting import plot_basins  # noqa: E402
from nfractals.polynomials import POLYNOMIALS, make_polynomial  # noqa: E402
from nfractals.runner import (  # noqa: E402
    CONVERGED,
    FAILED,
    NOT_CONVERGED,
    run_basins,
    single_point,
)

N = 1000
TOL = 1e-10
H_DAMPED = 0.7
DELTA = 1e-3
RADIUS = 0.2

f, df, d2f, roots = make_polynomial(POLYNOMIALS["z^3 - 2z + 2"])

x = np.linspace(-2, 2, N)
z0 = x[None, :] + 1j * x[:, None]
disk_0 = np.abs(z0 - 0) < RADIUS
disk_1 = np.abs(z0 - 1) < RADIUS

# (name, step, max_iter, initial state; None -> (z0,))
methods = [
    ("Newton", single_point(lambda z: newton_step(z, f, df)), 50, None),
    ("Halley", single_point(lambda z: halley_step(z, f, df, d2f)), 50, None),
    (f"Damped Newton (h={H_DAMPED})",
     single_point(lambda z: newton_step(z, f, df, h=H_DAMPED)), 200, None),
    ("Secant", lambda s: secant_step(s, f), 100, secant_init(z0, DELTA)),
    ("Müller", lambda s: muller_step(s, f), 100, muller_init(z0, DELTA)),
]

results = [
    run_basins(z0, step, f, roots, TOL, max_iter, init=init)
    for _, step, max_iter, init in methods
]

columns = [
    "method",
    "max_iter",
    "converged (%)",
    "not converged (%)",
    "failed (%)",
    f"not converged in disk r={RADIUS} centre 0 (%)",
    f"not converged in disk r={RADIUS} centre 1 (%)",
]

rows = []
for i, ((name, _, max_iter, _), (_, _, status)) in enumerate(zip(methods, results)):
    row = [
        name,
        max_iter,
        round(float(100 * np.mean(status == CONVERGED)), 4),
        round(float(100 * np.mean(status == NOT_CONVERGED)), 4),
        round(float(100 * np.mean(status == FAILED)), 4),
    ]
    if name == "Newton":
        for disk in (disk_0, disk_1):
            row.append(round(float(100 * np.mean(status[disk] == NOT_CONVERGED)), 4))
    else:
        row += ["", ""]
    rows.append(row)

# On-screen table.
texts = [
    [("-" if c == "" else f"{c:.2f}" if isinstance(c, float) else str(c)) for c in row]
    for row in rows
]
widths = [max(len(columns[i]), *(len(t[i]) for t in texts)) for i in range(len(columns))]
align = lambda row: "  ".join(
    c.ljust(widths[i]) if i == 0 else c.rjust(widths[i]) for i, c in enumerate(row)
)
print(align(columns))
print("  ".join("-" * w for w in widths))
for t in texts:
    print(align(t))

# CSV.
output_csv = PROJECT_ROOT / "results" / "e3_failures.csv"
with open(output_csv, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(columns)
    w.writerows(rows)
print(f"\nTable saved to {output_csv}")

# Figure: one map per method with a common iter_ref (99th percentile of the converged points of all).
iter_conv = np.concatenate([it[st == CONVERGED] for _, it, st in results])
iter_ref = np.percentile(iter_conv, 99) if iter_conv.size else None
print(f"Common iter_ref (99th percentile): {iter_ref}")

fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 6), constrained_layout=True)
for ax, (name, _, max_iter, _), (root, iters, status) in zip(axes, methods, results):
    plot_basins(root, iters, status, len(roots), ax=ax, iter_ref=iter_ref)
    ax.set_title(f"{name}, max_iter={max_iter}")

output_fig = PROJECT_ROOT / "figures" / "e3_failures.png"
fig.savefig(output_fig, dpi=100, bbox_inches="tight")
print(f"Figure saved to {output_fig}")
