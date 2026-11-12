import csv
import sys
from pathlib import Path

import numpy as np

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
from nfractals.runner import FALLO, NO_CONVERGE, run_basins, un_punto  # noqa: E402

TOL = 1e-10
H_AMORTIGUADO = 0.7
DELTA = 1e-3
TAMANOS = [250, 500, 1000, 2000]

f, df, d2f, roots = make_polynomial(POLYNOMIALS["z^3 - 1"])

# (nombre, paso, max_iter, fábrica del estado inicial a partir de z0; None -> (z0,))
metodos = [
    ("Newton", un_punto(lambda z: newton_step(z, f, df)), 50, None),
    ("Halley", un_punto(lambda z: halley_step(z, f, df, d2f)), 50, None),
    (f"Newton amortiguado (h={H_AMORTIGUADO})",
     un_punto(lambda z: newton_step(z, f, df, h=H_AMORTIGUADO)), 200, None),
    ("Secante", lambda s: secant_step(s, f), 100, lambda z: secant_init(z, DELTA)),
    ("Müller", lambda s: muller_step(s, f), 100, lambda z: muller_init(z, DELTA)),
]


def etiqueta(r):
    """Valor de la raíz como texto: '1', '-0.5+0.866j', ..."""
    re = round(r.real, 3) + 0.0
    im = round(r.imag, 3) + 0.0
    if im == 0:
        return f"{re:g}"
    return f"{re:g}{im:+g}j"


etiquetas = [etiqueta(r) for r in roots]
columnas = ["metodo", "ventana", "N"] + [f"raiz {e} (%)" for e in etiquetas] + [
    "fallo (%)",
    "no converge (%)",
]

filas = []
for nombre, step, max_iter, init in metodos:
    for n in TAMANOS:
        x = np.linspace(-2, 2, n)
        z0 = x[None, :] + 1j * x[:, None]
        raiz, _, estado = run_basins(
            z0, step, f, roots, TOL, max_iter, init=None if init is None else init(z0)
        )

        ventanas = [
            ("cuadrado", np.ones(z0.shape, dtype=bool)),
            ("disco", np.abs(z0) < 2),
        ]
        for ventana, mascara in ventanas:
            r_v = raiz[mascara]
            e_v = estado[mascara]
            pct = [100 * np.mean(r_v == k) for k in range(len(roots))]
            pct += [100 * np.mean(e_v == FALLO), 100 * np.mean(e_v == NO_CONVERGE)]
            filas.append([nombre, ventana, n] + [round(float(p), 4) for p in pct])
        print(f"calculado: {nombre}, N={n}", file=sys.stderr, flush=True)

# Tabla en pantalla.
textos = [[str(c) if not isinstance(c, float) else f"{c:.2f}" for c in fila] for fila in filas]
anchos = [max(len(columnas[i]), *(len(t[i]) for t in textos)) for i in range(len(columnas))]
alinear = lambda fila: "  ".join(
    c.ljust(anchos[i]) if i < 2 else c.rjust(anchos[i]) for i, c in enumerate(fila)
)
print(alinear(columnas))
print("  ".join("-" * a for a in anchos))
for t in textos:
    print(alinear(t))

# CSV.
salida = RAIZ_PROYECTO / "results" / "e1_basins.csv"
with open(salida, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(columnas)
    w.writerows(filas)
print(f"\nTabla guardada en {salida}")
