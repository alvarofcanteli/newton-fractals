import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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
    ACTIVE,
    CONVERGED,
    FAILED,
    NOT_CONVERGED,
    run_basins,
    single_point,
)

N = 1000
TOL = 1e-10
H_DAMPED = 0.7

f, df, d2f, roots = make_polynomial(POLYNOMIALS["z^3 - 1"])

x = np.linspace(-2, 2, N)
z0 = x[None, :] + 1j * x[:, None]

# (name, step, max_iter, init); init=None -> state (z0,)
methods = [
    ("Newton", single_point(lambda z: newton_step(z, f, df)), 50, None),
    ("Halley", single_point(lambda z: halley_step(z, f, df, d2f)), 50, None),
    (f"Damped Newton (h={H_DAMPED})",
     single_point(lambda z: newton_step(z, f, df, h=H_DAMPED)), 200, None),
    ("Secant", lambda s: secant_step(s, f), 100, secant_init(z0)),
    ("Müller", lambda s: muller_step(s, f), 100, muller_init(z0)),
]

STATUS_NAMES = {
    ACTIVE: "active",
    CONVERGED: "converged",
    FAILED: "failed",
    NOT_CONVERGED: "not converged",
}

fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 6), constrained_layout=True)

results = [
    run_basins(z0, step, roots, TOL, max_iter, init=init)
    for _, step, max_iter, init in methods
]

# Common iter_ref: 99th percentile of the iterations of the converged points of all methods together.
iter_ref = np.percentile(
    np.concatenate([it[st == CONVERGED] for _, it, st in results]), 99
)
print(f"Common iter_ref (99th percentile): {iter_ref:.2f}")

for ax, (name, _, max_iter, _), (root, iters, status) in zip(axes, methods, results):
    plot_basins(root, iters, status, len(roots), ax=ax, iter_ref=iter_ref)
    ax.set_title(f"{name}, max_iter={max_iter}")

    conv = status == CONVERGED
    print(f"\n== {name} (max_iter={max_iter}) ==")
    print("By root (% of all points):")
    for k, r in enumerate(roots):
        print(f"  Root {k} ({r.real:+.4f}{r.imag:+.4f}j): {np.mean(root == k):.2%}")
    print("By status (% of all points):")
    for code, text in STATUS_NAMES.items():
        print(f"  {text}: {np.mean(status == code):.2%}")
    if conv.any():
        print(f"Mean iterations (converged): {iters[conv].mean():.3f}")
    else:
        print("Mean iterations (converged): n/a")

output = PROJECT_ROOT / "figures" / "e0_compare.png"
fig.savefig(output, dpi=100, bbox_inches="tight")
print(f"\nFigure saved to {output}")
