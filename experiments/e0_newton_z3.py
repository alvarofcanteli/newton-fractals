from pathlib import Path

import numpy as np
from matplotlib.image import imsave

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]

from nfractals.methods import newton_step  # noqa: E402
from nfractals.plotting import plot_basins  # noqa: E402
from nfractals.polynomials import POLYNOMIALS, make_polynomial  # noqa: E402
from nfractals.runner import CONVERGIDO, run_basins, un_punto  # noqa: E402

N = 1000
TOL = 1e-10
MAX_ITER = 50

f, df, _, roots = make_polynomial(POLYNOMIALS["z^3 - 1"])

x = np.linspace(-2, 2, N)
z0 = x[None, :] + 1j * x[:, None]

raiz, iters, estado = run_basins(
    z0, un_punto(lambda z: newton_step(z, f, df)), f, roots, TOL, MAX_ITER
)

img = plot_basins(raiz, iters, estado, len(roots))

salida = RAIZ_PROYECTO / "figures" / "e0_newton_z3.png"
imsave(salida, img, origin="lower")
print(f"Imagen guardada en {salida}")

conv = estado == CONVERGIDO
print(f"Puntos convergidos: {conv.sum() / conv.size:.2%} del total")
for k, r in enumerate(roots):
    print(f"Raíz {k} ({r.real:+.4f}{r.imag:+.4f}j): {np.sum(raiz == k) / conv.sum():.2%}")
print(f"Iteraciones medias (convergidos): {iters[conv].mean():.3f}")
