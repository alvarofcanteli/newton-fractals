import csv
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
from nfractals.polynomials import POLYNOMIALS, make_polynomial, unique_roots  # noqa: E402
from nfractals.runner import (  # noqa: E402
    CONVERGED,
    FAILED,
    NOT_CONVERGED,
    run_basins,
    single_point,
)

N = 1000
H_DAMPED = 0.7
DELTA = 1e-3
MAX_ROOTS = 6

# (name in POLYNOMIALS, file name, tol)
POLYS = [
    ("z^3 - 1", "z3_minus_1", 1e-10),
    ("z^4 - 1", "z4_minus_1", 1e-10),
    ("z^5 - 1", "z5_minus_1", 1e-10),
    ("z^6 - 1", "z6_minus_1", 1e-10),
    ("(z-1)^2 (z+1)", "double_root", 1e-8),
    ("z^3 - 2z + 2", "z3_minus_2z_plus_2", 1e-10),
]

x = np.linspace(-2, 2, N)
z0 = x[None, :] + 1j * x[:, None]


def label(r):
    re = round(r.real, 3) + 0.0
    im = round(r.imag, 3) + 0.0
    return f"{re:g}" if im == 0 else f"{re:g}{im:+g}j"


columns = (
    ["polynomial", "method", "tol", "max_iter", "roots", "converged (%)",
     "not converged (%)", "failed (%)", "mean iter", "mean evals"]
    + [f"root {k + 1} (%)" for k in range(MAX_ROOTS)]
)
rows = []

for poly_name, file_name, tol in POLYS:
    f, df, d2f, roots_np = make_polynomial(POLYNOMIALS[poly_name])
    roots = unique_roots(roots_np)
    labels = [label(r) for r in roots]
    print(f"\n### {poly_name}: tol={tol:g}, {len(roots)} distinct roots "
          f"({len(roots_np)} from np.roots): {', '.join(labels)}", flush=True)

    # (name, step, max_iter, evals per step, initial evals, initial state)
    methods = [
        ("Newton", single_point(lambda z: newton_step(z, f, df)), 50, 2, 0, None),
        ("Halley", single_point(lambda z: halley_step(z, f, df, d2f)), 50, 3, 0, None),
        (f"Damped Newton (h={H_DAMPED})",
         single_point(lambda z: newton_step(z, f, df, h=H_DAMPED)), 200, 2, 0, None),
        ("Secant", lambda s: secant_step(s, f), 100, 1, 2, secant_init(z0, DELTA)),
        ("Müller", lambda s: muller_step(s, f), 100, 1, 3, muller_init(z0, DELTA)),
    ]
    results = [
        run_basins(z0, step, roots, tol, max_iter, init=init)
        for _, step, max_iter, _, _, init in methods
    ]

    for (name, _, max_iter, cost, ini, _), (root, iters, status) in zip(methods, results):
        conv = status == CONVERGED
        pct_root = [100 * np.mean(root == k) for k in range(len(roots))]
        it_mean = float(iters[conv].mean()) if conv.any() else float("nan")
        ev_mean = it_mean * cost + ini
        p_conv = 100 * np.mean(conv)
        p_nc = 100 * np.mean(status == NOT_CONVERGED)
        p_failed = 100 * np.mean(status == FAILED)
        rows.append(
            [poly_name, name, tol, max_iter, " | ".join(labels),
             round(p_conv, 4), round(p_nc, 4), round(p_failed, 4),
             round(it_mean, 3), round(ev_mean, 3)]
            + [round(float(p), 4) for p in pct_root]
            + [""] * (MAX_ROOTS - len(roots))
        )
        by_root = ", ".join(f"{e}: {p:.2f}%" for e, p in zip(labels, pct_root))
        print(f"- {poly_name} | {name} (max_iter={max_iter}): converged {p_conv:.2f}%, "
              f"not converged {p_nc:.2f}%, failed {p_failed:.2f}%, mean iter {it_mean:.3f}, "
              f"mean evals {ev_mean:.3f}; by root: {by_root}", flush=True)

    # Figure: five maps in a row with a common iter_ref within the polynomial.
    iter_conv = np.concatenate([it[st == CONVERGED] for _, it, st in results])
    iter_ref = np.percentile(iter_conv, 99) if iter_conv.size else None
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 5.4),
                             constrained_layout=True)
    for ax, (name, _, max_iter, *_), (root, iters, status) in zip(axes, methods, results):
        plot_basins(root, iters, status, len(roots), ax=ax, iter_ref=iter_ref)
        ax.set_title(f"{name}, max_iter={max_iter}", fontsize=9)
    fig.suptitle(f"{poly_name} (tol={tol:g}, iter_ref={iter_ref})")
    output_fig = PROJECT_ROOT / "figures" / f"e5_{file_name}.png"
    fig.savefig(output_fig, dpi=100, bbox_inches="tight")
    plt.close(fig)

output = PROJECT_ROOT / "results" / "e5_polynomials.csv"
with open(output, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(columns)
    w.writerows(rows)
print(f"\nTable saved to {output}")
