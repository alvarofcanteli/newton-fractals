"""Checks on E5: Newton on z^5 - 1 with max_iter = 40 and 50, and where the secant method
fails on z^4 - 1, z^5 - 1 and z^6 - 1 (inside and outside the disk |z0| < 1)."""
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from nfractals.methods import newton_step, secant_init, secant_step  # noqa: E402
from nfractals.polynomials import POLYNOMIALS, make_polynomial, unique_roots  # noqa: E402
from nfractals.runner import CONVERGED, FAILED, NOT_CONVERGED, run_basins, single_point  # noqa: E402

N = 1000
TOL = 1e-10

x = np.linspace(-2, 2, N)
z0 = x[None, :] + 1j * x[:, None]

f, df, d2f, r = make_polynomial(POLYNOMIALS["z^5 - 1"])
roots = unique_roots(r)
for mi in (40, 50):
    _, _, st = run_basins(z0, single_point(lambda z: newton_step(z, f, df)), roots, TOL, mi)
    print(f"Newton z^5-1 max_iter={mi}: not converged {np.mean(st == NOT_CONVERGED):.4%}, "
          f"failed {np.mean(st == FAILED):.4%}, unconverged (not converged + failed) "
          f"{np.mean(st != CONVERGED):.4%}")

inside = np.abs(z0) < 1
print(f"points |z0|<1: {inside.sum()} ({inside.mean():.2%}), >=1: {(~inside).sum()}")
for name in ("z^4 - 1", "z^5 - 1", "z^6 - 1"):
    f, df, d2f, r = make_polynomial(POLYNOMIALS[name])
    roots = unique_roots(r)
    _, _, st = run_basins(z0, lambda s: secant_step(s, f), roots, TOL, 100,
                          init=secant_init(z0, 1e-3))
    for label, m in (("not converged", st == NOT_CONVERGED), ("not converged+failed", st != CONVERGED)):
        tot = m.sum()
        print(f"Secant {name} [{label}]: |z0|<1: {m[inside].mean():.4%}; "
              f"|z0|>=1: {m[~inside].mean():.4%}; "
              f"fraction of the {tot} with |z0|<1: {(m & inside).sum() / tot:.4%}")
