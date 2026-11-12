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
from nfractals.polynomials import POLYNOMIALS, make_polynomial, unique_roots  # noqa: E402
from nfractals.runner import (  # noqa: E402
    CONVERGIDO,
    FALLO,
    NO_CONVERGE,
    run_basins,
    un_punto,
)

N = 1000
H_AMORTIGUADO = 0.7
DELTA = 1e-3
MAX_RAICES = 6

# (nombre en POLYNOMIALS, nombre de archivo, tol)
POLINOMIOS = [
    ("z^3 - 1", "z3_menos_1", 1e-10),
    ("z^4 - 1", "z4_menos_1", 1e-10),
    ("z^5 - 1", "z5_menos_1", 1e-10),
    ("z^6 - 1", "z6_menos_1", 1e-10),
    ("(z-1)^2 (z+1)", "raiz_doble", 1e-8),
    ("z^3 - 2z + 2", "z3_menos_2z_mas_2", 1e-10),
]

x = np.linspace(-2, 2, N)
z0 = x[None, :] + 1j * x[:, None]


def etiqueta(r):
    re = round(r.real, 3) + 0.0
    im = round(r.imag, 3) + 0.0
    return f"{re:g}" if im == 0 else f"{re:g}{im:+g}j"


columnas = (
    ["polinomio", "metodo", "tol", "max_iter", "raices", "convergido (%)",
     "no converge (%)", "fallo (%)", "iter media", "evals media"]
    + [f"raiz {k + 1} (%)" for k in range(MAX_RAICES)]
)
filas = []

for nombre_pol, archivo, tol in POLINOMIOS:
    f, df, d2f, roots_np = make_polynomial(POLYNOMIALS[nombre_pol])
    roots = unique_roots(roots_np)
    etiquetas = [etiqueta(r) for r in roots]
    print(f"\n### {nombre_pol}: tol={tol:g}, {len(roots)} raíces distintas "
          f"({len(roots_np)} de np.roots): {', '.join(etiquetas)}", flush=True)

    # (nombre, paso, max_iter, evals por paso, evals iniciales, estado inicial)
    metodos = [
        ("Newton", un_punto(lambda z: newton_step(z, f, df)), 50, 2, 0, None),
        ("Halley", un_punto(lambda z: halley_step(z, f, df, d2f)), 50, 3, 0, None),
        (f"Newton amortiguado (h={H_AMORTIGUADO})",
         un_punto(lambda z: newton_step(z, f, df, h=H_AMORTIGUADO)), 200, 2, 0, None),
        ("Secante", lambda s: secant_step(s, f), 100, 1, 2, secant_init(z0, DELTA)),
        ("Müller", lambda s: muller_step(s, f), 100, 1, 3, muller_init(z0, DELTA)),
    ]
    resultados = [
        run_basins(z0, step, f, roots, tol, max_iter, init=init)
        for _, step, max_iter, _, _, init in metodos
    ]

    for (nombre, _, max_iter, coste, ini, _), (raiz, iters, estado) in zip(metodos, resultados):
        conv = estado == CONVERGIDO
        pct_raiz = [100 * np.mean(raiz == k) for k in range(len(roots))]
        it_media = float(iters[conv].mean()) if conv.any() else float("nan")
        ev_media = it_media * coste + ini
        p_conv = 100 * np.mean(conv)
        p_nc = 100 * np.mean(estado == NO_CONVERGE)
        p_fallo = 100 * np.mean(estado == FALLO)
        filas.append(
            [nombre_pol, nombre, tol, max_iter, " | ".join(etiquetas),
             round(p_conv, 4), round(p_nc, 4), round(p_fallo, 4),
             round(it_media, 3), round(ev_media, 3)]
            + [round(float(p), 4) for p in pct_raiz]
            + [""] * (MAX_RAICES - len(roots))
        )
        por_raiz = ", ".join(f"{e}: {p:.2f}%" for e, p in zip(etiquetas, pct_raiz))
        print(f"- {nombre_pol} | {nombre} (max_iter={max_iter}): convergido {p_conv:.2f}%, "
              f"no converge {p_nc:.2f}%, fallo {p_fallo:.2f}%, iter media {it_media:.3f}, "
              f"evals media {ev_media:.3f}; por raíz: {por_raiz}", flush=True)

    # Figura: cinco mapas en fila con iter_ref común dentro del polinomio.
    iter_conv = np.concatenate([it[est == CONVERGIDO] for _, it, est in resultados])
    iter_ref = np.percentile(iter_conv, 99) if iter_conv.size else None
    fig, axes = plt.subplots(1, len(metodos), figsize=(5 * len(metodos), 5.4),
                             constrained_layout=True)
    for ax, (nombre, _, max_iter, *_), (raiz, iters, estado) in zip(axes, metodos, resultados):
        plot_basins(raiz, iters, estado, len(roots), ax=ax, iter_ref=iter_ref)
        ax.set_title(f"{nombre}, max_iter={max_iter}", fontsize=9)
    fig.suptitle(f"{nombre_pol} (tol={tol:g}, iter_ref={iter_ref})")
    salida_fig = RAIZ_PROYECTO / "figures" / f"e5_{archivo}.png"
    fig.savefig(salida_fig, dpi=100, bbox_inches="tight")
    plt.close(fig)

salida = RAIZ_PROYECTO / "results" / "e5_polynomials.csv"
with open(salida, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(columnas)
    w.writerows(filas)
print(f"\nTabla guardada en {salida}")
