"""Comprobaciones sobre E5: Newton en z^5 - 1 con max_iter = 40 y 50, y dónde falla la
secante en z^4 - 1, z^5 - 1 y z^6 - 1 (dentro y fuera del disco |z0| < 1)."""
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from nfractals.methods import newton_step, secant_init, secant_step  # noqa: E402
from nfractals.polynomials import POLYNOMIALS, make_polynomial, unique_roots  # noqa: E402
from nfractals.runner import CONVERGIDO, FALLO, NO_CONVERGE, run_basins, un_punto  # noqa: E402

N = 1000
TOL = 1e-10

x = np.linspace(-2, 2, N)
z0 = x[None, :] + 1j * x[:, None]

f, df, d2f, r = make_polynomial(POLYNOMIALS["z^5 - 1"])
roots = unique_roots(r)
for mi in (40, 50):
    _, _, e = run_basins(z0, un_punto(lambda z: newton_step(z, f, df)), f, roots, TOL, mi)
    print(f"Newton z^5-1 max_iter={mi}: no converge {np.mean(e == NO_CONVERGE):.4%}, "
          f"fallo {np.mean(e == FALLO):.4%}, sin converger (no converge + fallo) "
          f"{np.mean(e != CONVERGIDO):.4%}")

dentro = np.abs(z0) < 1
print(f"puntos |z0|<1: {dentro.sum()} ({dentro.mean():.2%}), >=1: {(~dentro).sum()}")
for nom in ("z^4 - 1", "z^5 - 1", "z^6 - 1"):
    f, df, d2f, r = make_polynomial(POLYNOMIALS[nom])
    roots = unique_roots(r)
    _, _, e = run_basins(z0, lambda s: secant_step(s, f), f, roots, TOL, 100,
                         init=secant_init(z0, 1e-3))
    for etiq, m in (("no converge", e == NO_CONVERGE), ("no converge+fallo", e != CONVERGIDO)):
        tot = m.sum()
        print(f"Secante {nom} [{etiq}]: |z0|<1: {m[dentro].mean():.4%}; "
              f"|z0|>=1: {m[~dentro].mean():.4%}; "
              f"fracción de los {tot} con |z0|<1: {(m & dentro).sum() / tot:.4%}")
