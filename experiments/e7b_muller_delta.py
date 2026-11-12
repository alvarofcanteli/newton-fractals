"""Müller en E7: efecto de delta en la diferencia complex64 / complex128 y distancia de los
píxeles que difieren a las fronteras.

Genera figures/e7_diff_muller.png (delta = 1e-3).
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]

from nfractals.methods import muller_init, muller_step  # noqa: E402
from nfractals.polynomials import POLYNOMIALS, make_polynomial, unique_roots  # noqa: E402
from nfractals.runner import CONVERGIDO, run_basins  # noqa: E402

N = 1000
TOL = {np.complex64: 1e-5, np.complex128: 1e-10}

# Misma malla que en E7: una vez en complex128, redondeada a complex64 y devuelta a complex128.
x = np.linspace(-2, 2, N)
z128 = x[None, :] + 1j * x[:, None]
z64 = z128.astype(np.complex64)
zd = z64.astype(np.complex128)


def dilatar(mascara, r):
    """Dilatación con ventana cuadrada de lado 2r+1 (distancia de Chebyshev <= r)."""
    out = mascara
    for eje in (0, 1):
        acum = out.copy()
        for s in range(1, r + 1):
            a = [slice(None), slice(None)]
            b = [slice(None), slice(None)]
            # Desplazamiento hacia índices mayores.
            a[eje], b[eje] = slice(s, None), slice(None, -s)
            acum[tuple(a)] |= out[tuple(b)]
            # Desplazamiento hacia índices menores.
            a[eje], b[eje] = slice(None, -s), slice(s, None)
            acum[tuple(a)] |= out[tuple(b)]
        out = acum
    return out


def cerca(raiz, ok, r):
    """Píxeles convergidos a distancia de Chebyshev <= r de otro píxel convergido con
    raíz distinta."""
    resultado = np.zeros(raiz.shape, bool)
    for k in np.unique(raiz[ok]):
        otros = ok & (raiz != k)
        resultado |= (raiz == k) & ok & dilatar(otros, r)
    return resultado


def corre(delta):
    out = {}
    for dt, z in ((np.complex64, z64), (np.complex128, zd)):
        f, _, _, rr = make_polynomial(POLYNOMIALS["z^3 - 1"], dtype=dt)
        out[dt] = run_basins(z, lambda s: muller_step(s, f), f, unique_roots(rr), TOL[dt], 100,
                             init=muller_init(z, delta))
    return out


for delta in (1e-1, 1e-2, 1e-3):
    o = corre(delta)
    r64, i64, e64 = o[np.complex64]
    r128, i128, e128 = o[np.complex128]
    c64, c128 = e64 == CONVERGIDO, e128 == CONVERGIDO
    dif = c64 & c128 & (r64 != r128)
    solo = c64 ^ c128
    print(f"delta={delta:g}: raíz difiere {100 * dif.mean():.4f}% ({dif.sum()} px); "
          f"convergido solo en una: {solo.sum()} px; conv. c64 {100 * c64.mean():.3f}%, "
          f"c128 {100 * c128.mean():.3f}%; iter media c64 {i64[c64].mean():.3f}, "
          f"c128 {i128[c128].mean():.3f}")
    radios = (3,) if delta != 1e-3 else (3, 10, 20)
    for r in radios:
        fr = cerca(r128, c128, r)
        pct = 100 * fr[dif].mean() if dif.any() else float("nan")
        print(f"   ≤{r} px: {pct:.2f}% de los que difieren; {100 * fr.mean():.2f}% de todos los píxeles")
    if delta == 1e-3:
        fr3 = cerca(r128, c128, 3)
        mapa = np.zeros(r128.shape, np.uint8)
        mapa[fr3] = 1
        mapa[solo] = 3
        mapa[dif] = 2
        fig, ax = plt.subplots(figsize=(9, 9), constrained_layout=True)
        ax.imshow(mapa, origin="lower", interpolation="nearest",
                  cmap=ListedColormap(["white", "#dddddd", "#d62728", "#1f77b4"]), vmin=0, vmax=3)
        ax.set_axis_off()
        ax.set_title(f"Müller δ=1e-3, z^3 - 1: complex64 vs complex128 ({dif.sum()} píxeles con raíz distinta)\n"
                     "gris: a ≤ 3 px de otra raíz; rojo: raíz distinta; azul: converge solo en una",
                     fontsize=10)
        salida = RAIZ_PROYECTO / "figures" / "e7_diff_muller.png"
        fig.savefig(salida, dpi=110, bbox_inches="tight")
        print(f"Figura guardada en {salida}")
