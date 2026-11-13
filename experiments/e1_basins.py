import csv
import sys
from pathlib import Path

import numpy as np

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
from nfractals.runner import FAILED, NOT_CONVERGED, run_basins, single_point  # noqa: E402

TOL = 1e-10
H_DAMPED = 0.7
DELTA = 1e-3
SIZES = [250, 500, 1000, 2000]

f, df, d2f, roots = make_polynomial(POLYNOMIALS["z^3 - 1"])

# (name, step, max_iter, factory of the initial state from z0; None -> (z0,))
methods = [
    ("Newton", single_point(lambda z: newton_step(z, f, df)), 50, None),
    ("Halley", single_point(lambda z: halley_step(z, f, df, d2f)), 50, None),
    (f"Damped Newton (h={H_DAMPED})",
     single_point(lambda z: newton_step(z, f, df, h=H_DAMPED)), 200, None),
    ("Secant", lambda s: secant_step(s, f), 100, lambda z: secant_init(z, DELTA)),
    ("Müller", lambda s: muller_step(s, f), 100, lambda z: muller_init(z, DELTA)),
]


def label(r):
    """Value of the root as text: '1', '-0.5+0.866j', ..."""
    re = round(r.real, 3) + 0.0
    im = round(r.imag, 3) + 0.0
    if im == 0:
        return f"{re:g}"
    return f"{re:g}{im:+g}j"


labels = [label(r) for r in roots]
columns = ["method", "window", "N"] + [f"root {e} (%)" for e in labels] + [
    "failed (%)",
    "not converged (%)",
]

rows = []
for name, step, max_iter, init in methods:
    for n in SIZES:
        x = np.linspace(-2, 2, n)
        z0 = x[None, :] + 1j * x[:, None]
        root, _, status = run_basins(
            z0, step, roots, TOL, max_iter, init=None if init is None else init(z0)
        )

        windows = [
            ("square", np.ones(z0.shape, dtype=bool)),
            ("disk", np.abs(z0) < 2),
        ]
        for window, mask in windows:
            r_w = root[mask]
            s_w = status[mask]
            pct = [100 * np.mean(r_w == k) for k in range(len(roots))]
            pct += [100 * np.mean(s_w == FAILED), 100 * np.mean(s_w == NOT_CONVERGED)]
            rows.append([name, window, n] + [round(float(p), 4) for p in pct])
        print(f"computed: {name}, N={n}", file=sys.stderr, flush=True)

# On-screen table.
texts = [[str(c) if not isinstance(c, float) else f"{c:.2f}" for c in row] for row in rows]
widths = [max(len(columns[i]), *(len(t[i]) for t in texts)) for i in range(len(columns))]
align = lambda row: "  ".join(
    c.ljust(widths[i]) if i < 2 else c.rjust(widths[i]) for i, c in enumerate(row)
)
print(align(columns))
print("  ".join("-" * w for w in widths))
for t in texts:
    print(align(t))

# CSV.
output = PROJECT_ROOT / "results" / "e1_basins.csv"
with open(output, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(columns)
    w.writerows(rows)
print(f"\nTable saved to {output}")
