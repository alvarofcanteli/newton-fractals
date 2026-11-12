import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]

from nfractals.methods import newton_step  # noqa: E402
from nfractals.plotting import plot_basins  # noqa: E402
from nfractals.polynomials import POLYNOMIALS, make_polynomial, unique_roots  # noqa: E402
from nfractals.runner import CONVERGIDO, FALLO, NO_CONVERGE, run_basins, un_punto  # noqa: E402

N = 1000
TOL = 1e-10
MAX_ITER = 300
COSTE = 2  # evaluaciones por paso (f y f')
HS = [0.3, 0.5, 0.7, 0.9, 1.0]
POLINOMIOS = [("z^3 - 1", "z3_menos_1"), ("z^3 - 2z + 2", "z3_menos_2z_mas_2")]

x = np.linspace(-2, 2, N)
z0 = x[None, :] + 1j * x[:, None]


def fraccion_frontera(raiz, estado):
    """Fracción de píxeles convergidos con algún vecino (4-conectividad) convergido
    a otra raíz. Los pares en que algún píxel no converge no cuentan."""
    ok = estado == CONVERGIDO
    marca = np.zeros(raiz.shape, dtype=bool)
    # Vecinos verticales.
    d = ok[1:, :] & ok[:-1, :] & (raiz[1:, :] != raiz[:-1, :])
    marca[1:, :] |= d
    marca[:-1, :] |= d
    # Vecinos horizontales.
    d = ok[:, 1:] & ok[:, :-1] & (raiz[:, 1:] != raiz[:, :-1])
    marca[:, 1:] |= d
    marca[:, :-1] |= d
    return marca.mean()  # sobre todos los píxeles de la malla


def etiqueta(r):
    re = round(r.real, 3) + 0.0
    im = round(r.imag, 3) + 0.0
    return f"{re:g}" if im == 0 else f"{re:g}{im:+g}j"


MAX_RAICES = 3
columnas = (
    ["polinomio", "h", "max_iter", "iter media", "iter mediana", "evals media",
     "no converge (%)", "fallo (%)", "frontera (%)", "raices"]
    + [f"raiz {k + 1} (%)" for k in range(MAX_RAICES)]
)
filas = []
curvas = {}

print(f"Frontera: fracción de píxeles de la malla convergidos que tienen algún vecino "
      f"(4-conectividad) convergido a otra raíz. tol={TOL:g}, max_iter={MAX_ITER}, "
      f"coste {COSTE} evals/paso.")

for nombre_pol, archivo in POLINOMIOS:
    f, df, d2f, r = make_polynomial(POLYNOMIALS[nombre_pol])
    roots = unique_roots(r)
    etiquetas = [etiqueta(q) for q in roots]
    print(f"\n### {nombre_pol}: raíces {', '.join(etiquetas)}", flush=True)

    resultados = []
    for h in HS:
        resultados.append(run_basins(
            z0, un_punto(lambda z, h=h: newton_step(z, f, df, h=h)),
            f, roots, TOL, MAX_ITER,
        ))

    iter_conv = np.concatenate([it[est == CONVERGIDO] for _, it, est in resultados])
    iter_ref = np.percentile(iter_conv, 99)

    medias, fronteras = [], []
    for h, (raiz, iters, estado) in zip(HS, resultados):
        conv = estado == CONVERGIDO
        it = iters[conv]
        it_media, it_med = float(it.mean()), float(np.median(it))
        pct_raiz = [100 * np.mean(raiz == k) for k in range(len(roots))]
        p_nc = 100 * np.mean(estado == NO_CONVERGE)
        p_fallo = 100 * np.mean(estado == FALLO)
        fr = 100 * fraccion_frontera(raiz, estado)
        medias.append(it_media)
        fronteras.append(fr)
        filas.append(
            [nombre_pol, h, MAX_ITER, round(it_media, 3), round(it_med, 3),
             round(it_media * COSTE, 3), round(p_nc, 4), round(p_fallo, 4), round(fr, 4),
             " | ".join(etiquetas)]
            + [round(float(p), 4) for p in pct_raiz]
            + [""] * (MAX_RAICES - len(roots))
        )
        por_raiz = ", ".join(f"{e}: {p:.2f}%" for e, p in zip(etiquetas, pct_raiz))
        print(f"- {nombre_pol} | h={h:g}: iter media {it_media:.3f}, mediana {it_med:.1f}, "
              f"evals media {it_media * COSTE:.3f}; no converge {p_nc:.2f}%, fallo {p_fallo:.2f}%; "
              f"frontera {fr:.2f}%; por raíz: {por_raiz}", flush=True)
    curvas[nombre_pol] = (medias, fronteras)

    fig, axes = plt.subplots(1, len(HS), figsize=(5 * len(HS), 5.4), constrained_layout=True)
    for ax, h, (raiz, iters, estado) in zip(axes, HS, resultados):
        plot_basins(raiz, iters, estado, len(roots), ax=ax, iter_ref=iter_ref)
        ax.set_title(f"h = {h:g}", fontsize=10)
    fig.suptitle(f"Newton amortiguado en {nombre_pol} (tol={TOL:g}, max_iter={MAX_ITER}, "
                 f"iter_ref={iter_ref:.1f})")
    salida_fig = RAIZ_PROYECTO / "figures" / f"e6_maps_{archivo}.png"
    fig.savefig(salida_fig, dpi=100, bbox_inches="tight")
    plt.close(fig)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
for nombre_pol, (medias, fronteras) in curvas.items():
    ax1.plot(HS, medias, marker="o", label=nombre_pol)
    ax2.plot(HS, fronteras, marker="o", label=nombre_pol)
ax1.set_xlabel("h")
ax1.set_ylabel("iteraciones medias (convergidos)")
ax2.set_xlabel("h")
ax2.set_ylabel("fracción de frontera (%)")
for ax in (ax1, ax2):
    ax.set_xticks(HS)
    ax.grid(alpha=0.3)
    ax.legend()
salida_curvas = RAIZ_PROYECTO / "figures" / "e6_curves.png"
fig.savefig(salida_curvas, dpi=100, bbox_inches="tight")
plt.close(fig)

salida = RAIZ_PROYECTO / "results" / "e6_damped.csv"
with open(salida, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(columnas)
    w.writerows(filas)
print(f"\nTabla guardada en {salida}")
