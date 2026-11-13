"""E5c: effect of the tolerance on the double root (z-1)^2 (z+1) under the criterion |z - r| < tol.

Same setting as E5 (complex128, window [-2, 2]^2, N = 1000, delta = 1e-3, same max_iter) for
tol in {1e-4, 1e-6, 1e-8, 1e-10}. Near a double root the attainable accuracy of z is about
sqrt(machine epsilon), so small tolerances sit at the numerical floor.

Generates results/e5c_double_root_tol.csv. The median distance to the nearest root is reported
for the points that do not converge: for NOT_CONVERGED the distance of the last iterate, for FAILED
the distance of the last finite iterate (the final value is not finite).
"""
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
from nfractals.polynomials import POLYNOMIALS, make_polynomial, unique_roots  # noqa: E402
from nfractals.runner import CONVERGED, FAILED, NOT_CONVERGED, run_basins, single_point  # noqa: E402

N = 1000
H_DAMPED = 0.7
DELTA = 1e-3
TOLS = (1e-4, 1e-6, 1e-8, 1e-10)
POLY = "(z-1)^2 (z+1)"

x = np.linspace(-2, 2, N)
z0 = x[None, :] + 1j * x[:, None]

f, df, d2f, roots_np = make_polynomial(POLYNOMIALS[POLY])
roots = unique_roots(roots_np)

# (name, step, max_iter, evals per step, initial evals, initial state); same as E5.
methods = [
    ("Newton", single_point(lambda z: newton_step(z, f, df)), 50, 2, 0, None),
    ("Halley", single_point(lambda z: halley_step(z, f, df, d2f)), 50, 3, 0, None),
    (f"Damped Newton (h={H_DAMPED})",
     single_point(lambda z: newton_step(z, f, df, h=H_DAMPED)), 200, 2, 0, None),
    ("Secant", lambda s: secant_step(s, f), 100, 1, 2, secant_init(z0, DELTA)),
    ("Müller", lambda s: muller_step(s, f), 100, 1, 3, muller_init(z0, DELTA)),
]


def distance_to_roots(z):
    return np.abs(z[:, None] - roots[None, :]).min(axis=1)


def median_distance(step, init, mask, max_iter):
    """Median distance to the nearest root of the points in `mask`: last iterate after max_iter
    steps, or the last finite iterate if the value becomes non-finite. NaN if mask is empty."""
    if not mask.any():
        return float("nan")
    state = tuple(a[mask].copy() for a in (init if init is not None else (z0,)))
    last = state[-1].copy()
    for _ in range(max_iter):
        with np.errstate(all="ignore"):
            state = step(state)
        z = state[-1]
        last = np.where(np.isfinite(z), z, last)
    return float(np.median(distance_to_roots(last)))


columns = ["polynomial", "method", "tol", "max_iter", "converged (%)", "not converged (%)",
           "failed (%)", "mean iter", "median distance to root of not converged points",
           "median distance to root of failed points (last finite iterate)"]
rows = []

print(f"{POLY}: {len(roots)} distinct roots {np.round(roots, 6).tolist()}, N = {N}, delta = {DELTA:g}")
for tol in TOLS:
    print(f"\n### tol = {tol:g}")
    for name, step, max_iter, _, _, init in methods:
        _, iters, status = run_basins(z0, step, roots, tol, max_iter, init=init)
        conv = status == CONVERGED
        nc, fl = status == NOT_CONVERGED, status == FAILED
        it_mean = float(iters[conv].mean()) if conv.any() else float("nan")
        d_nc = median_distance(step, init, nc, max_iter) if nc.any() else float("nan")
        d_fl = median_distance(step, init, fl, max_iter) if fl.any() else float("nan")
        rows.append([POLY, name, tol, max_iter, round(100 * conv.mean(), 4), round(100 * nc.mean(), 4),
                     round(100 * fl.mean(), 4), round(it_mean, 3),
                     "" if np.isnan(d_nc) else f"{d_nc:.3e}", "" if np.isnan(d_fl) else f"{d_fl:.3e}"])
        print(f"- {name} (max_iter={max_iter}): converged {100 * conv.mean():.4f}%, "
              f"not converged {100 * nc.mean():.4f}% ({nc.sum()} px), failed {100 * fl.mean():.4f}% ({fl.sum()} px), "
              f"mean iter {it_mean:.3f}; median distance to root: not converged {d_nc:.3e}, "
              f"failed (last finite) {d_fl:.3e}", flush=True)

output = PROJECT_ROOT / "results" / "e5c_double_root_tol.csv"
with open(output, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(columns)
    w.writerows(rows)
print("\nTable saved to results/e5c_double_root_tol.csv")
