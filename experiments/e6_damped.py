import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from nfractals.methods import newton_step  # noqa: E402
from nfractals.metrics import boundary_fraction  # noqa: E402
from nfractals.plotting import plot_basins, poly_latex  # noqa: E402
from nfractals.polynomials import POLYNOMIALS, make_polynomial, unique_roots  # noqa: E402
from nfractals.runner import CONVERGED, FAILED, NOT_CONVERGED, run_basins, single_point  # noqa: E402

N = 1000
TOL = 1e-10
MAX_ITER = 300
COST = 2  # evaluations per step (f and f')
HS = [0.3, 0.5, 0.7, 0.9, 1.0]
POLYS = [("z^3 - 1", "z3_minus_1"), ("z^3 - 2z + 2", "z3_minus_2z_plus_2")]

x = np.linspace(-2, 2, N)
z0 = x[None, :] + 1j * x[:, None]


def label(r):
    re = round(r.real, 3) + 0.0
    im = round(r.imag, 3) + 0.0
    return f"{re:g}" if im == 0 else f"{re:g}{im:+g}j"


MAX_ROOTS = 3
columns = (
    ["polynomial", "h", "max_iter", "mean iter", "median iter", "mean evals",
     "not converged (%)", "failed (%)", "boundary (%)", "roots"]
    + [f"root {k + 1} (%)" for k in range(MAX_ROOTS)]
)
rows = []
curves = {}

print(f"Boundary: fraction of grid pixels that are converged and have some converged neighbour "
      f"(4-connectivity) with another root. tol={TOL:g}, max_iter={MAX_ITER}, "
      f"cost {COST} evals/step.")

for poly_name, file_name in POLYS:
    f, df, d2f, r = make_polynomial(POLYNOMIALS[poly_name])
    roots = unique_roots(r)
    labels = [label(q) for q in roots]
    print(f"\n### {poly_name}: roots {', '.join(labels)}", flush=True)

    results = []
    for h in HS:
        results.append(run_basins(
            z0, single_point(lambda z, h=h: newton_step(z, f, df, h=h)),
            roots, TOL, MAX_ITER,
        ))

    iter_conv = np.concatenate([it[st == CONVERGED] for _, it, st in results])
    iter_ref = np.percentile(iter_conv, 99)

    means, boundaries = [], []
    for h, (root, iters, status) in zip(HS, results):
        conv = status == CONVERGED
        it = iters[conv]
        it_mean, it_med = float(it.mean()), float(np.median(it))
        pct_root = [100 * np.mean(root == k) for k in range(len(roots))]
        p_nc = 100 * np.mean(status == NOT_CONVERGED)
        p_failed = 100 * np.mean(status == FAILED)
        fr = 100 * boundary_fraction(root, status)
        means.append(it_mean)
        boundaries.append(fr)
        rows.append(
            [poly_name, h, MAX_ITER, round(it_mean, 3), round(it_med, 3),
             round(it_mean * COST, 3), round(p_nc, 4), round(p_failed, 4), round(fr, 4),
             " | ".join(labels)]
            + [round(float(p), 4) for p in pct_root]
            + [""] * (MAX_ROOTS - len(roots))
        )
        by_root = ", ".join(f"{e}: {p:.2f}%" for e, p in zip(labels, pct_root))
        print(f"- {poly_name} | h={h:g}: mean iter {it_mean:.3f}, median {it_med:.1f}, "
              f"mean evals {it_mean * COST:.3f}; not converged {p_nc:.2f}%, failed {p_failed:.2f}%; "
              f"boundary {fr:.2f}%; by root: {by_root}", flush=True)
    curves[poly_name] = (means, boundaries)

    fig, axes = plt.subplots(1, len(HS), figsize=(5 * len(HS), 5.4), constrained_layout=True)
    for ax, h, (root, iters, status) in zip(axes, HS, results):
        plot_basins(root, iters, status, len(roots), ax=ax, iter_ref=iter_ref)
        ax.set_title(f"h = {h:g}", fontsize=10)
    fig.suptitle(f"Damped Newton on {poly_latex(poly_name)} (tol={TOL:g}, max_iter={MAX_ITER}, "
                 f"iter_ref={iter_ref:.1f})")
    output_fig = PROJECT_ROOT / "figures" / f"e6_maps_{file_name}.png"
    fig.savefig(output_fig, dpi=100, bbox_inches="tight")
    plt.close(fig)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
for poly_name, (means, boundaries) in curves.items():
    ax1.plot(HS, means, marker="o", label=poly_latex(poly_name))
    ax2.plot(HS, boundaries, marker="o", label=poly_latex(poly_name))
ax1.set_xlabel("h")
ax1.set_ylabel("mean iterations (converged)")
ax2.set_xlabel("h")
ax2.set_ylabel("boundary fraction (%)")
for ax in (ax1, ax2):
    ax.set_xticks(HS)
    ax.grid(alpha=0.3)
    ax.legend()
output_curves = PROJECT_ROOT / "figures" / "e6_curves.png"
fig.savefig(output_curves, dpi=100, bbox_inches="tight")
plt.close(fig)

output = PROJECT_ROOT / "results" / "e6_damped.csv"
with open(output, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(columns)
    w.writerows(rows)
print(f"\nTable saved to {output}")
