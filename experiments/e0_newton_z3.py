from pathlib import Path

import numpy as np
from matplotlib.image import imsave

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from nfractals.methods import newton_step  # noqa: E402
from nfractals.plotting import plot_basins  # noqa: E402
from nfractals.polynomials import POLYNOMIALS, make_polynomial  # noqa: E402
from nfractals.runner import CONVERGED, run_basins, single_point  # noqa: E402

N = 1000
TOL = 1e-10
MAX_ITER = 50

f, df, _, roots = make_polynomial(POLYNOMIALS["z^3 - 1"])

x = np.linspace(-2, 2, N)
z0 = x[None, :] + 1j * x[:, None]

root, iters, status = run_basins(
    z0, single_point(lambda z: newton_step(z, f, df)), roots, TOL, MAX_ITER
)

img = plot_basins(root, iters, status, len(roots))

output = PROJECT_ROOT / "figures" / "e0_newton_z3.png"
imsave(output, img, origin="lower")
print(f"Image saved to {output}")

conv = status == CONVERGED
print(f"Converged points: {conv.sum() / conv.size:.2%} of the total")
for k, r in enumerate(roots):
    print(f"Root {k} ({r.real:+.4f}{r.imag:+.4f}j): {np.sum(root == k) / conv.sum():.2%}")
print(f"Mean iterations (converged): {iters[conv].mean():.3f}")
