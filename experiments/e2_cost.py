import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

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
from nfractals.polynomials import POLYNOMIALS, make_polynomial  # noqa: E402
from nfractals.runner import CONVERGIDO, run_basins, un_punto  # noqa: E402

N = 1000
TOL = 1e-10
H_AMORTIGUADO = 0.7
DELTA = 1e-3

f, df, d2f, roots = make_polynomial(POLYNOMIALS["z^3 - 1"])

x = np.linspace(-2, 2, N)
z0 = x[None, :] + 1j * x[:, None]

# (nombre, paso, max_iter, evaluaciones por paso, evaluaciones iniciales, estado inicial)
# evals por punto = iteraciones * evals por paso + evals iniciales.
metodos = [
    ("Newton", un_punto(lambda z: newton_step(z, f, df)), 50, 2, 0, None),
    ("Halley", un_punto(lambda z: halley_step(z, f, df, d2f)), 50, 3, 0, None),
    (f"Newton amortiguado (h={H_AMORTIGUADO})",
     un_punto(lambda z: newton_step(z, f, df, h=H_AMORTIGUADO)), 200, 2, 0, None),
    ("Secante", lambda s: secant_step(s, f), 100, 1, 2, secant_init(z0, DELTA)),
    ("Müller", lambda s: muller_step(s, f), 100, 1, 3, muller_init(z0, DELTA)),
]

columnas = [
    "metodo",
    "max_iter",
    "evals por paso",
    "evals iniciales",
    "iter media",
    "iter mediana",
    "evals media",
    "evals mediana",
]

filas = []
iteraciones = []
for nombre, step, max_iter, coste, ini, init in metodos:
    _, iters, estado = run_basins(z0, step, f, roots, TOL, max_iter, init=init)
    it = iters[estado == CONVERGIDO]
    iteraciones.append(it)
    filas.append([
        nombre,
        max_iter,
        coste,
        ini,
        round(float(it.mean()), 3),
        round(float(np.median(it)), 3),
        round(float(it.mean() * coste + ini), 3),
        round(float(np.median(it) * coste + ini), 3),
    ])

# Tabla en pantalla.
textos = [[str(c) if not isinstance(c, float) else f"{c:.3f}" for c in fila] for fila in filas]
anchos = [max(len(columnas[i]), *(len(t[i]) for t in textos)) for i in range(len(columnas))]
alinear = lambda fila: "  ".join(
    c.ljust(anchos[i]) if i == 0 else c.rjust(anchos[i]) for i, c in enumerate(fila)
)
print(alinear(columnas))
print("  ".join("-" * a for a in anchos))
for t in textos:
    print(alinear(t))

# CSV.
salida_csv = RAIZ_PROYECTO / "results" / "e2_cost.csv"
with open(salida_csv, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(columnas)
    w.writerows(filas)
print(f"\nTabla guardada en {salida_csv}")

# Histogramas con la misma escala en el eje x.
it_max = max(int(it.max()) for it in iteraciones)
bins = np.arange(1, it_max + 2) - 0.5

fig, axes = plt.subplots(
    len(metodos), 1, figsize=(8, 3 * len(metodos)), sharex=True, constrained_layout=True
)
for ax, (nombre, _, max_iter, *_), it in zip(axes, metodos, iteraciones):
    ax.hist(it, bins=bins)
    ax.set_title(f"{nombre}, max_iter={max_iter}")
    ax.set_ylabel("puntos")
axes[-1].set_xlabel("iteraciones hasta converger")
axes[-1].set_xlim(0, it_max + 1)

salida_fig = RAIZ_PROYECTO / "figures" / "e2_hist.png"
fig.savefig(salida_fig, dpi=100, bbox_inches="tight")
print(f"Figura guardada en {salida_fig}")
