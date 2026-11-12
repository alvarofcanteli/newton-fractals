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
from nfractals.plotting import plot_basins  # noqa: E402
from nfractals.polynomials import POLYNOMIALS, make_polynomial  # noqa: E402
from nfractals.runner import CONVERGIDO, FALLO, NO_CONVERGE, run_basins, un_punto  # noqa: E402

N = 1000
TOL = 1e-10
H_AMORTIGUADO = 0.7
DELTA = 1e-3
RADIO = 0.2

f, df, d2f, roots = make_polynomial(POLYNOMIALS["z^3 - 2z + 2"])

x = np.linspace(-2, 2, N)
z0 = x[None, :] + 1j * x[:, None]
disco_0 = np.abs(z0 - 0) < RADIO
disco_1 = np.abs(z0 - 1) < RADIO

# (nombre, paso, max_iter, estado inicial; None -> (z0,))
metodos = [
    ("Newton", un_punto(lambda z: newton_step(z, f, df)), 50, None),
    ("Halley", un_punto(lambda z: halley_step(z, f, df, d2f)), 50, None),
    (f"Newton amortiguado (h={H_AMORTIGUADO})",
     un_punto(lambda z: newton_step(z, f, df, h=H_AMORTIGUADO)), 200, None),
    ("Secante", lambda s: secant_step(s, f), 100, secant_init(z0, DELTA)),
    ("Müller", lambda s: muller_step(s, f), 100, muller_init(z0, DELTA)),
]

resultados = [
    run_basins(z0, step, f, roots, TOL, max_iter, init=init)
    for _, step, max_iter, init in metodos
]

columnas = [
    "metodo",
    "max_iter",
    "convergido (%)",
    "no converge (%)",
    "fallo (%)",
    f"no converge en disco r={RADIO} centro 0 (%)",
    f"no converge en disco r={RADIO} centro 1 (%)",
]

filas = []
for i, ((nombre, _, max_iter, _), (_, _, estado)) in enumerate(zip(metodos, resultados)):
    fila = [
        nombre,
        max_iter,
        round(float(100 * np.mean(estado == CONVERGIDO)), 4),
        round(float(100 * np.mean(estado == NO_CONVERGE)), 4),
        round(float(100 * np.mean(estado == FALLO)), 4),
    ]
    if nombre == "Newton":
        for disco in (disco_0, disco_1):
            fila.append(round(float(100 * np.mean(estado[disco] == NO_CONVERGE)), 4))
    else:
        fila += ["", ""]
    filas.append(fila)

# Tabla en pantalla.
textos = [
    [("-" if c == "" else f"{c:.2f}" if isinstance(c, float) else str(c)) for c in fila]
    for fila in filas
]
anchos = [max(len(columnas[i]), *(len(t[i]) for t in textos)) for i in range(len(columnas))]
alinear = lambda fila: "  ".join(
    c.ljust(anchos[i]) if i == 0 else c.rjust(anchos[i]) for i, c in enumerate(fila)
)
print(alinear(columnas))
print("  ".join("-" * a for a in anchos))
for t in textos:
    print(alinear(t))

# CSV.
salida_csv = RAIZ_PROYECTO / "results" / "e3_failures.csv"
with open(salida_csv, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(columnas)
    w.writerows(filas)
print(f"\nTabla guardada en {salida_csv}")

# Figura: un mapa por método con iter_ref común (percentil 99 de los convergidos de todos).
iter_conv = np.concatenate([it[est == CONVERGIDO] for _, it, est in resultados])
iter_ref = np.percentile(iter_conv, 99) if iter_conv.size else None
print(f"iter_ref común (percentil 99): {iter_ref}")

fig, axes = plt.subplots(1, len(metodos), figsize=(6 * len(metodos), 6), constrained_layout=True)
for ax, (nombre, _, max_iter, _), (raiz, iters, estado) in zip(axes, metodos, resultados):
    plot_basins(raiz, iters, estado, len(roots), ax=ax, iter_ref=iter_ref)
    ax.set_title(f"{nombre}, max_iter={max_iter}")

salida_fig = RAIZ_PROYECTO / "figures" / "e3_failures.png"
fig.savefig(salida_fig, dpi=100, bbox_inches="tight")
print(f"Figura guardada en {salida_fig}")
