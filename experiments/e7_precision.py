import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]

from nfractals.methods import (  # noqa: E402
    halley_step,
    muller_init,
    muller_step,
    newton_step,
    secant_init,
    secant_step,
)
from nfractals.polynomials import POLYNOMIALS, make_polynomial, unique_roots  # noqa: E402
from nfractals.runner import CONVERGIDO, run_basins, un_punto  # noqa: E402

N = 1000
H_AMORTIGUADO = 0.7
DELTA = 1e-3
RADIO_FRONTERA = 3  # píxeles, distancia de Chebyshev
EPS = {np.complex64: float(np.finfo(np.float32).eps), np.complex128: float(np.finfo(np.float64).eps)}
TOL = {np.complex64: 1e-5, np.complex128: 1e-10}

# Malla: una sola vez en complex128, redondeada a complex64 y devuelta a complex128,
# de modo que ambas ejecuciones parten exactamente de los mismos puntos.
x = np.linspace(-2, 2, N)
z0_c128 = x[None, :] + 1j * x[:, None]
z0_c64 = z0_c128.astype(np.complex64)
z0_doble = z0_c64.astype(np.complex128)
mallas = {np.complex64: z0_c64, np.complex128: z0_doble}

print("Tolerancias y épsilon de máquina:")
for dt in (np.complex64, np.complex128):
    print(f"  {np.dtype(dt).name}: eps = {EPS[dt]:.2e}, tol = {TOL[dt]:.0e} = {TOL[dt] / EPS[dt]:.1e} eps. "
          "La evaluación de f junto a una raíz simple tiene un suelo de ruido de unos pocos eps; "
          "tol queda muy por encima de ese suelo (para no declarar 'no converge' por ruido de "
          "redondeo) y muy por debajo de la escala de f en la malla (O(1)).")

coef = POLYNOMIALS["z^3 - 1"]
sistemas = {}
for dt in (np.complex64, np.complex128):
    f, df, d2f, r = make_polynomial(coef, dtype=dt)
    sistemas[dt] = (f, df, d2f, unique_roots(r))


def metodos_para(dt):
    f, df, d2f, _ = sistemas[dt]
    z0 = mallas[dt]
    # (nombre, paso, max_iter, estado inicial)
    return [
        ("Newton", un_punto(lambda z: newton_step(z, f, df)), 50, None),
        ("Halley", un_punto(lambda z: halley_step(z, f, df, d2f)), 50, None),
        (f"Newton amortiguado (h={H_AMORTIGUADO})",
         un_punto(lambda z: newton_step(z, f, df, h=H_AMORTIGUADO)), 200, None),
        ("Secante", lambda s: secant_step(s, f), 100, secant_init(z0, DELTA)),
        ("Müller", lambda s: muller_step(s, f), 100, muller_init(z0, DELTA)),
    ]


def cerca_de_otra_raiz(raiz, ok, radio):
    """Píxeles convergidos a distancia de Chebyshev <= radio de algún píxel convergido
    con raíz distinta."""
    h, w = raiz.shape
    cerca = np.zeros(raiz.shape, dtype=bool)
    for dy in range(-radio, radio + 1):
        for dx in range(-radio, radio + 1):
            if dy == 0 and dx == 0:
                continue
            a = (slice(max(0, dy), h + min(0, dy)), slice(max(0, dx), w + min(0, dx)))
            b = (slice(max(0, -dy), h + min(0, -dy)), slice(max(0, -dx), w + min(0, -dx)))
            cerca[a] |= ok[a] & ok[b] & (raiz[a] != raiz[b])
    return cerca


columnas = [
    "metodo", "max_iter", "tol complex64", "tol complex128",
    "raiz difiere (% de todos)", "raiz difiere (% de convergidos en ambas)",
    "convergido solo en complex64 (%)", "convergido solo en complex128 (%)",
    "iter media complex64", "iter media complex128",
    "cerca de frontera: % de todos los pixeles",
    "cerca de frontera: % de los que difieren en raiz",
    "cerca de frontera: % de los que difieren en algo",
]
filas = []
res = {dt: [run_basins(mallas[dt], paso, sistemas[dt][0], sistemas[dt][3], TOL[dt], mi, init=init)
            for _, paso, mi, init in metodos_para(dt)]
       for dt in (np.complex64, np.complex128)}

print("\nResultados (z^3 - 1, N = 1000, mismos puntos de partida):")
for i, (nombre, _, max_iter, _) in enumerate(metodos_para(np.complex128)):
    raiz64, it64, est64 = res[np.complex64][i]
    raiz128, it128, est128 = res[np.complex128][i]
    c64, c128 = est64 == CONVERGIDO, est128 == CONVERGIDO
    ambas = c64 & c128
    dif_raiz = ambas & (raiz64 != raiz128)
    solo64, solo128 = c64 & ~c128, c128 & ~c64
    dif_algo = dif_raiz | solo64 | solo128

    frontera = cerca_de_otra_raiz(raiz128, c128, RADIO_FRONTERA)
    p_front_todos = 100 * frontera.mean()
    p_front_raiz = 100 * frontera[dif_raiz].mean() if dif_raiz.any() else float("nan")
    p_front_algo = 100 * frontera[dif_algo].mean() if dif_algo.any() else float("nan")

    p_dif_todos = 100 * dif_raiz.mean()
    p_dif_ambas = 100 * dif_raiz.sum() / ambas.sum()
    i64, i128 = float(it64[c64].mean()), float(it128[c128].mean())
    filas.append([nombre, max_iter, TOL[np.complex64], TOL[np.complex128],
                  round(p_dif_todos, 5), round(p_dif_ambas, 5),
                  round(100 * solo64.mean(), 5), round(100 * solo128.mean(), 5),
                  round(i64, 3), round(i128, 3), round(p_front_todos, 4),
                  round(p_front_raiz, 4), round(p_front_algo, 4)])
    print(f"- {nombre} (max_iter={max_iter}): raíz difiere en {p_dif_todos:.4f}% de los píxeles "
          f"({dif_raiz.sum()} píxeles; {p_dif_ambas:.4f}% de los convergidos en ambas); "
          f"convergido solo en complex64 {100 * solo64.mean():.4f}% ({solo64.sum()}), "
          f"solo en complex128 {100 * solo128.mean():.4f}% ({solo128.sum()}); "
          f"iter media complex64 {i64:.3f}, complex128 {i128:.3f}; "
          f"cerca de la frontera (≤{RADIO_FRONTERA} px): {p_front_raiz:.2f}% de los que difieren en raíz, "
          f"{p_front_algo:.2f}% de los que difieren en algo, frente a {p_front_todos:.2f}% de todos los píxeles",
          flush=True)

    if nombre == "Newton":
        mapa = np.zeros(raiz128.shape, dtype=np.uint8)  # 0 fondo, 1 cerca de frontera,
        mapa[frontera] = 1                              # 2 raíz distinta, 3 solo en una
        mapa[solo64 | solo128] = 3
        mapa[dif_raiz] = 2
        fig, ax = plt.subplots(figsize=(9, 9), constrained_layout=True)
        ax.imshow(mapa, origin="lower", interpolation="nearest",
                  cmap=ListedColormap(["white", "#dddddd", "#d62728", "#1f77b4"]), vmin=0, vmax=3)
        ax.set_axis_off()
        ax.set_title(f"Newton, z^3 - 1: complex64 vs complex128 ({dif_raiz.sum()} píxeles con raíz distinta, "
                     f"{(solo64 | solo128).sum()} convergidos solo en una)\n"
                     f"gris: a ≤ {RADIO_FRONTERA} px de otra raíz; rojo: raíz distinta; azul: converge solo en una",
                     fontsize=10)
        salida_fig = RAIZ_PROYECTO / "figures" / "e7_diff_newton.png"
        fig.savefig(salida_fig, dpi=110, bbox_inches="tight")
        plt.close(fig)

salida = RAIZ_PROYECTO / "results" / "e7_precision.csv"
with open(salida, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(columnas)
    w.writerows(filas)
print(f"\nTabla guardada en {salida}")
