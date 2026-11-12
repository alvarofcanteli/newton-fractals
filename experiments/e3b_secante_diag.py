"""Diagnóstico de la secante en z^3 - 1: no convergidos según max_iter y su estado final.

Genera figures/diag_secante.png (mapa con max_iter = 400) y barre delta con max_iter = 200.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]

from nfractals.methods import secant_init, secant_step  # noqa: E402
from nfractals.plotting import plot_basins  # noqa: E402
from nfractals.polynomials import POLYNOMIALS, make_polynomial  # noqa: E402
from nfractals.runner import CONVERGIDO, FALLO, NO_CONVERGE, run_basins  # noqa: E402

N = 1000
TOL = 1e-10
ITER_REF = 31.0  # el mismo iter_ref que e0_compare con los cuatro primeros métodos

f, df, d2f, roots = make_polynomial(POLYNOMIALS["z^3 - 1"])
x = np.linspace(-2, 2, N)
z0 = x[None, :] + 1j * x[:, None]

res = {}
for mi in (100, 200, 400):
    r = run_basins(z0, lambda s: secant_step(s, f), f, roots, TOL, mi, init=secant_init(z0))
    res[mi] = r
    est = r[2]
    print(f"max_iter={mi}: no converge {np.mean(est == NO_CONVERGE):.4%} "
          f"({(est == NO_CONVERGE).sum()}), fallo {np.mean(est == FALLO):.4%}, "
          f"convergido {np.mean(est == CONVERGIDO):.4%}")

# Estado final de los no convergidos con max_iter = 400 (bucle propio con la misma lógica).
S = tuple(a.copy() for a in secant_init(z0))
activo = np.ones(z0.shape, bool)
for _ in range(400):
    if not activo.any():
        break
    nuevo = secant_step(tuple(a[activo] for a in S), f)
    for a, n in zip(S, nuevo):
        a[activo] = n
    zn = nuevo[-1]
    with np.errstate(all="ignore"):
        fin = np.isfinite(zn)
        terminado = ~fin | (fin & (np.abs(f(zn)) < TOL))
    activo.flat[np.flatnonzero(activo)[terminado]] = False
nc = activo
print("coincide con run_basins:", np.array_equal(nc, res[400][2] == NO_CONVERGE))

z = S[-1][nc]
a = np.abs(z)
print(f"no convergidos: {z.size}")
print(f"  |z| finito: {np.isfinite(a).sum()}")
print(f"  |z| > 100: {(a > 100).sum()}")
print(f"  |z| < 3: {(a < 3).sum()}")
print(f"  3 <= |z| <= 100: {((a >= 3) & (a <= 100)).sum()}")
if z.size:
    print(f"  |z| min/mediana/max: {a.min():.3g} / {np.median(a):.3g} / {a.max():.3g}")
    zz = z[a < 3]
    if zz.size:
        d = np.abs(zz[:, None] - roots[None, :]).min(1)
        print(f"  de los |z|<3: distancia mínima a una raíz, mediana {np.median(d):.3g}, "
              f"máx {d.max():.3g}; |f| mediana {np.median(np.abs(f(zz))):.3g}")
    print(f"  |z0| mediana de los no convergidos: {np.median(np.abs(z0[nc])):.3g}")

fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
raiz, iters, est = res[400]
plot_basins(raiz, iters, est, len(roots), ax=ax, iter_ref=ITER_REF)
ax.set_title("Secante, z^3 - 1, max_iter=400")
salida = RAIZ_PROYECTO / "figures" / "diag_secante.png"
fig.savefig(salida, dpi=100, bbox_inches="tight")
print(f"Figura guardada en {salida}")

# Barrido de delta (z_{-1} = z_0 + delta) con max_iter = 200.
print("\nNo convergidos con max_iter=200 según delta:")
for delta in (1e-2, 1e-3, 1e-4, 1e-6):
    e = run_basins(z0, lambda s: secant_step(s, f), f, roots, TOL, 200,
                   init=secant_init(z0, delta))[2]
    print(f"  delta={delta:g}: no converge {np.mean(e == NO_CONVERGE):.4%} "
          f"({(e == NO_CONVERGE).sum()})")
