"""Diagnosis of the secant method on z^3 - 1: non-converged points according to max_iter
and their final state.

Generates figures/diag_secant.png (map with max_iter = 400) and sweeps delta with max_iter = 200.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from nfractals.methods import secant_init, secant_step  # noqa: E402
from nfractals.plotting import plot_basins, poly_latex  # noqa: E402
from nfractals.polynomials import POLYNOMIALS, make_polynomial  # noqa: E402
from nfractals.runner import CONVERGED, FAILED, NOT_CONVERGED, run_basins  # noqa: E402

N = 1000
TOL = 1e-10
ITER_REF = 30.0  # the same iter_ref as e0_compare with the first four methods

f, df, d2f, roots = make_polynomial(POLYNOMIALS["z^3 - 1"])
x = np.linspace(-2, 2, N)
z0 = x[None, :] + 1j * x[:, None]

res = {}
for mi in (100, 200, 400):
    r = run_basins(z0, lambda s: secant_step(s, f), roots, TOL, mi, init=secant_init(z0))
    res[mi] = r
    st = r[2]
    print(f"max_iter={mi}: not converged {np.mean(st == NOT_CONVERGED):.4%} "
          f"({(st == NOT_CONVERGED).sum()}), failed {np.mean(st == FAILED):.4%}, "
          f"converged {np.mean(st == CONVERGED):.4%}")

# Final state of the points not converged with max_iter = 400 (own loop with the same logic).
state = tuple(a.copy() for a in secant_init(z0))
active = np.ones(z0.shape, bool)
for _ in range(400):
    if not active.any():
        break
    new = secant_step(tuple(a[active] for a in state), f)
    for a, n in zip(state, new):
        a[active] = n
    zn = new[-1]
    with np.errstate(all="ignore"):
        fin = np.isfinite(zn)
        dist_min = np.full(zn.shape, np.inf)
        dist_min[fin] = np.abs(zn[fin][:, None] - roots[None, :]).min(axis=1)
        done = ~fin | (dist_min < TOL)
    active.flat[np.flatnonzero(active)[done]] = False
nc = active
print("matches run_basins:", np.array_equal(nc, res[400][2] == NOT_CONVERGED))

z = state[-1][nc]
a = np.abs(z)
print(f"not converged: {z.size}")
print(f"  |z| finite: {np.isfinite(a).sum()}")
print(f"  |z| > 100: {(a > 100).sum()}")
print(f"  |z| < 3: {(a < 3).sum()}")
print(f"  3 <= |z| <= 100: {((a >= 3) & (a <= 100)).sum()}")
if z.size:
    print(f"  |z| min/median/max: {a.min():.3g} / {np.median(a):.3g} / {a.max():.3g}")
    zz = z[a < 3]
    if zz.size:
        d = np.abs(zz[:, None] - roots[None, :]).min(1)
        print(f"  of those with |z|<3: minimum distance to a root, median {np.median(d):.3g}, "
              f"max {d.max():.3g}; median |f| {np.median(np.abs(f(zz))):.3g}")
    print(f"  median |z0| of the non-converged points: {np.median(np.abs(z0[nc])):.3g}")

fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
root, iters, st = res[400]
plot_basins(root, iters, st, len(roots), ax=ax, iter_ref=ITER_REF)
ax.set_title(f"Secant, {poly_latex('z^3 - 1')}, max_iter=400")
output = PROJECT_ROOT / "figures" / "diag_secant.png"
fig.savefig(output, dpi=100, bbox_inches="tight")
print(f"Figure saved to {output}")

# Sweep of delta (z_{-1} = z_0 + delta) with max_iter = 200.
print("\nNot converged with max_iter=200 according to delta:")
for delta in (1e-2, 1e-3, 1e-4, 1e-6):
    st = run_basins(z0, lambda s: secant_step(s, f), roots, TOL, 200,
                    init=secant_init(z0, delta))[2]
    print(f"  delta={delta:g}: not converged {np.mean(st == NOT_CONVERGED):.4%} "
          f"({(st == NOT_CONVERGED).sum()})")
