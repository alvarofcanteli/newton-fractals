"""E10: the comparison of E5 on two non-polynomial functions, sin z and e^z - z.

This is an EXTENSION beyond the polynomial scope of the course guide. Same setting as E5: N = 1000,
tol = 1e-10, delta = 1e-3, same max_iter and criterion |z - r| < tol, with r the nearest reference root.

- sin z, window [-5, 5]^2: roots k*pi; the nearest root is pi * round(Re z / pi) (exact), root id k.
- e^z - z, window [-3, 3]^2: infinitely many roots z_k = -W_k(-1); the reference roots are those with
  |k| <= 20 (mpmath, cast to complex128). A point that never gets within tol of a listed root but ends
  with |f(z)| < tol is counted separately as "converged to an unlisted root".

Overflow (e^z with large Re z, sin/cos with large |Im z|) gives non-finite values: FAILED.
Generates results/e10_non_polynomials.csv and figures/e10_*.png.
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from nfractals.functions import make_functions  # noqa: E402
from nfractals.methods import (  # noqa: E402
    halley_step,
    muller_init,
    muller_step,
    newton_step,
    secant_init,
    secant_step,
)
from nfractals.metrics import boundary_fraction  # noqa: E402
from nfractals.plotting import plot_basins  # noqa: E402
from nfractals.runner import (  # noqa: E402
    CONVERGED,
    FAILED,
    NOT_CONVERGED,
    run_basins,
    single_point,
)

N = 1000
TOL = 1e-10
H_DAMPED = 0.7
DELTA = 1e-3
MAX_GROUPS = 6
FILE_NAMES = {"sin z": "sin_z", "e^z - z": "exp_z_minus_z"}

columns = (
    ["function", "method", "tol", "max_iter", "converged (%)",
     "not converged (%) (excluding unlisted roots)", "failed (%)",
     "converged to an unlisted root (%)", "mean iter", "median iter", "mean evals", "boundary (%)",
     "root groups"]
    + [f"group {k + 1} (%)" for k in range(MAX_GROUPS)]
)
rows = []

for fn in make_functions():
    f, df, d2f = fn.f, fn.df, fn.d2f
    x = np.linspace(-fn.half_width, fn.half_width, N)
    z0 = x[None, :] + 1j * x[:, None]
    n_groups = len(fn.group_labels)
    print(f"\n### {fn.name}: window [-{fn.half_width:g}, {fn.half_width:g}]^2, groups: "
          f"{', '.join(fn.group_labels)}", flush=True)

    # (name, step, max_iter, evals per step, initial evals, initial state)
    methods = [
        ("Newton", single_point(lambda z: newton_step(z, f, df)), 50, 2, 0, None),
        ("Halley", single_point(lambda z: halley_step(z, f, df, d2f)), 50, 3, 0, None),
        (f"Damped Newton (h={H_DAMPED})",
         single_point(lambda z: newton_step(z, f, df, h=H_DAMPED)), 200, 2, 0, None),
        ("Secant", lambda s: secant_step(s, f), 100, 1, 2, secant_init(z0, DELTA)),
        ("Müller", lambda s: muller_step(s, f), 100, 1, 3, muller_init(z0, DELTA)),
    ]
    results = []
    for _, step, max_iter, _, _, init in methods:
        with np.errstate(all="ignore"):
            root, iters, status, z_final = run_basins(
                z0, step, fn.nearest, TOL, max_iter, init=init, return_final=True)
            unlisted = (status == NOT_CONVERGED) & (np.abs(f(z_final)) < TOL)
        results.append((root, iters, status, unlisted))

    for (name, _, max_iter, cost, ini, _), (root, iters, status, unlisted) in zip(methods, results):
        conv = status == CONVERGED
        group = np.where(conv, fn.group_of(root), -1)
        pct_group = [100 * np.mean(group == k) for k in range(n_groups)]
        it_mean = float(iters[conv].mean()) if conv.any() else float("nan")
        it_med = float(np.median(iters[conv])) if conv.any() else float("nan")
        p_conv = 100 * np.mean(conv)
        p_nc = 100 * np.mean((status == NOT_CONVERGED) & ~unlisted)
        p_failed = 100 * np.mean(status == FAILED)
        p_unl = 100 * np.mean(unlisted)
        p_bd = 100 * boundary_fraction(root, status)
        rows.append(
            [fn.name, name, TOL, max_iter, round(p_conv, 4), round(p_nc, 4), round(p_failed, 4),
             round(p_unl, 4), round(it_mean, 3), round(it_med, 3), round(it_mean * cost + ini, 3),
             round(p_bd, 4), " | ".join(fn.group_labels)]
            + [round(float(p), 4) for p in pct_group]
            + [""] * (MAX_GROUPS - n_groups)
        )
        by_group = ", ".join(f"{e}: {p:.2f}%" for e, p in zip(fn.group_labels, pct_group))
        print(f"- {fn.name} | {name} (max_iter={max_iter}): converged {p_conv:.2f}%, "
              f"not converged {p_nc:.2f}%, failed {p_failed:.2f}%, unlisted root {p_unl:.2f}%, "
              f"mean iter {it_mean:.3f}, median {it_med:.1f}, mean evals {it_mean * cost + ini:.3f}, "
              f"boundary {p_bd:.3f}%; by group: {by_group}", flush=True)

    # Figure: five maps in a row with a common iter_ref within the function.
    iter_conv = np.concatenate([it[st == CONVERGED] for _, it, st, _ in results])
    iter_ref = np.percentile(iter_conv, 99) if iter_conv.size else None
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 5.4),
                             constrained_layout=True)
    for ax, (name, _, max_iter, *_), (root, iters, status, _) in zip(axes, methods, results):
        group = np.where(status == CONVERGED, fn.group_of(root), 0)
        plot_basins(group, iters, status, n_groups, ax=ax, iter_ref=iter_ref)
        ax.set_title(f"{name}, max_iter={max_iter}", fontsize=9)
    w = f"{fn.half_width:g}"
    fig.suptitle(f"{fn.latex} on [−{w}, {w}]², N = {N} (tol={TOL:g}, iter_ref={iter_ref})")
    fig.savefig(PROJECT_ROOT / "figures" / f"e10_{FILE_NAMES[fn.name]}.png", dpi=100,
                bbox_inches="tight")
    plt.close(fig)

output = PROJECT_ROOT / "results" / "e10_non_polynomials.csv"
with open(output, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(columns)
    w.writerows(rows)
print(f"\nTable saved to {output}")
