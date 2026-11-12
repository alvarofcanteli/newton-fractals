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
from nfractals.runner import (  # noqa: E402
    ACTIVO,
    CONVERGIDO,
    FALLO,
    NO_CONVERGE,
    run_basins,
    un_punto,
)

N = 1000
TOL = 1e-10
H_AMORTIGUADO = 0.7

f, df, d2f, roots = make_polynomial(POLYNOMIALS["z^3 - 1"])

x = np.linspace(-2, 2, N)
z0 = x[None, :] + 1j * x[:, None]

# (nombre, step, max_iter, init); init=None -> estado (z0,)
metodos = [
    ("Newton", un_punto(lambda z: newton_step(z, f, df)), 50, None),
    ("Halley", un_punto(lambda z: halley_step(z, f, df, d2f)), 50, None),
    (f"Newton amortiguado (h={H_AMORTIGUADO})",
     un_punto(lambda z: newton_step(z, f, df, h=H_AMORTIGUADO)), 200, None),
    ("Secante", lambda s: secant_step(s, f), 100, secant_init(z0)),
    ("Müller", lambda s: muller_step(s, f), 100, muller_init(z0)),
]

NOMBRES_ESTADO = {
    ACTIVO: "activo",
    CONVERGIDO: "convergido",
    FALLO: "fallo",
    NO_CONVERGE: "no converge",
}

fig, axes = plt.subplots(1, len(metodos), figsize=(6 * len(metodos), 6), constrained_layout=True
)

resultados = [
    run_basins(z0, step, f, roots, TOL, max_iter, init=init)
    for _, step, max_iter, init in metodos
]

# iter_ref común: percentil 99 de las iteraciones de los convergidos de todos los métodos juntos.
iter_ref = np.percentile(
    np.concatenate([it[est == CONVERGIDO] for _, it, est in resultados]), 99
)
print(f"iter_ref común (percentil 99): {iter_ref:.2f}")

for ax, (nombre, _, max_iter, _), (raiz, iters, estado) in zip(axes, metodos, resultados):
    plot_basins(raiz, iters, estado, len(roots), ax=ax, iter_ref=iter_ref)
    ax.set_title(f"{nombre}, max_iter={max_iter}")

    conv = estado == CONVERGIDO
    print(f"\n== {nombre} (max_iter={max_iter}) ==")
    print("Por raíz (% de los puntos totales):")
    for k, r in enumerate(roots):
        print(f"  Raíz {k} ({r.real:+.4f}{r.imag:+.4f}j): {np.mean(raiz == k):.2%}")
    print("Por estado (% de los puntos totales):")
    for codigo, texto in NOMBRES_ESTADO.items():
        print(f"  {texto}: {np.mean(estado == codigo):.2%}")
    if conv.any():
        print(f"Iteraciones medias (convergidos): {iters[conv].mean():.3f}")
    else:
        print("Iteraciones medias (convergidos): n/a")

salida = RAIZ_PROYECTO / "figures" / "e0_compare.png"
fig.savefig(salida, dpi=100, bbox_inches="tight")
print(f"\nFigura guardada en {salida}")
