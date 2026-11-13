from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from nfractals.methods import newton_step  # noqa: E402
from nfractals.plotting import plot_basins, poly_latex  # noqa: E402
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

conv = status == CONVERGED
iter_ref = np.percentile(iters[conv], 99)  # same default as plot_basins

fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
plot_basins(root, iters, status, len(roots), ax=ax, iter_ref=iter_ref)
ax.set_title(f"Newton, {poly_latex('z^3 - 1')} on [−2, 2]², N = {N} "
             f"(tol={TOL:g}, max_iter={MAX_ITER}, iter_ref={iter_ref:g})", fontsize=10)

output = PROJECT_ROOT / "figures" / "e0_newton_z3.png"
fig.savefig(output, dpi=100, bbox_inches="tight")
plt.close(fig)
print(f"Figure saved to {output}")

print(f"Converged points: {conv.sum() / conv.size:.2%} of the total")
for k, r in enumerate(roots):
    print(f"Root {k} ({r.real:+.4f}{r.imag:+.4f}j): {np.sum(root == k) / conv.sum():.2%}")
print(f"Mean iterations (converged): {iters[conv].mean():.3f}")
