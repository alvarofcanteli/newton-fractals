"""E12: reliability profile by starting radius |z0|.

Polynomials z^3 - 1, z^3 - 2z + 2, z^6 - 1 and the random degree-5 polynomial (seed 2026), the five
methods, baseline setting of E2 (window [-2, 2]^2, N = 1000, complex128, tol = 1e-10, delta = 1e-3, the
default max_iter of each method: Newton 50, Halley 50, damped Newton (h = 0.7) 200, secant 100,
Muller 100). The grid is split into rings by |z0|: [0, 0.5), [0.5, 1), [1, 1.5), [1.5, 2) and
[2, 2 sqrt(2)] (the last one is the corners of the square, outside the inscribed circle; the rings
have different numbers of pixels). Per ring: pixels, % converged, % not converged, % failed and the
mean evaluations of the converged points (NaN if none).

Generates results/e12_radius_profile.csv and figures/e12_radius_profile.png.
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

from nfractals.methods import (  # noqa: E402
    halley_step,
    muller_init,
    muller_step,
    newton_step,
    secant_init,
    secant_step,
)
from nfractals.plotting import poly_latex  # noqa: E402
from nfractals.polynomials import POLYNOMIALS, make_polynomial, unique_roots  # noqa: E402
from nfractals.runner import CONVERGED, FAILED, NOT_CONVERGED, run_basins, single_point  # noqa: E402

POLYS = ["z^3 - 1", "z^3 - 2z + 2", "z^6 - 1", "random degree 5"]
N = 1000
TOL = 1e-10
H_DAMPED = 0.7
DELTA = 1e-3
EDGES = [0.0, 0.5, 1.0, 1.5, 2.0]  # the last ring is [2, 2 sqrt(2)]
RING_LABELS = ["[0, 0.5)", "[0.5, 1)", "[1, 1.5)", "[1.5, 2)", "[2, 2√2] (corners)"]


def rings_of(z0):
    """Ring index (0..4) of every pixel by |z0|: [0,0.5), [0.5,1), [1,1.5), [1.5,2), [2,2*sqrt(2)]."""
    return np.digitize(np.abs(z0), EDGES[1:])


def methods_for(f, df, d2f, z0):
    # (name, step, max_iter, evals per step, initial evals, initial state)
    return [
        ("Newton", single_point(lambda z: newton_step(z, f, df)), 50, 2, 0, None),
        ("Halley", single_point(lambda z: halley_step(z, f, df, d2f)), 50, 3, 0, None),
        (f"Damped Newton (h={H_DAMPED})",
         single_point(lambda z: newton_step(z, f, df, h=H_DAMPED)), 200, 2, 0, None),
        ("Secant", lambda s: secant_step(s, f), 100, 1, 2, secant_init(z0, DELTA)),
        ("Müller", lambda s: muller_step(s, f), 100, 1, 3, muller_init(z0, DELTA)),
    ]


def ring_table(poly_name, n=N):
    """Rows (polynomial, method, ring, pixels, % converged, % not converged, % failed, mean evals)
    for the baseline run at grid size n, plus, for each method, the exact ring-weighted numbers
    {method: (pixels, % converged, mean evals)} obtained from the integer counts and iteration sums of
    the rings (no rounding), and the same numbers computed directly on the whole grid."""
    f, df, d2f, roots_np = make_polynomial(POLYNOMIALS[poly_name])
    roots = unique_roots(roots_np)
    x = np.linspace(-2, 2, n)
    z0 = x[None, :] + 1j * x[:, None]
    ring = rings_of(z0)
    rows, global_numbers, weighted = [], {}, {}
    for name, step, max_iter, cost, ini, init in methods_for(f, df, d2f, z0):
        _, iters, status = run_basins(z0, step, roots, TOL, max_iter, init=init)
        conv = status == CONVERGED
        global_numbers[name] = (
            100 * float(np.mean(conv)),
            float(iters[conv].mean()) * cost + ini if conv.any() else float("nan"),
        )
        n_px = n_conv = it_sum = 0
        for k, label in enumerate(RING_LABELS):
            m = ring == k
            c = conv & m
            n_px += int(m.sum())
            n_conv += int(c.sum())
            it_sum += int(iters[c].sum())
            mean_evals = float(iters[c].mean()) * cost + ini if c.any() else float("nan")
            rows.append([poly_name, name, label, int(m.sum()),
                         round(100 * float(np.mean(conv[m])), 4),
                         round(100 * float(np.mean((status == NOT_CONVERGED)[m])), 4),
                         round(100 * float(np.mean((status == FAILED)[m])), 4),
                         "nan" if np.isnan(mean_evals) else round(mean_evals, 3)])
        weighted[name] = (n_px, 100 * n_conv / n_px, it_sum / n_conv * cost + ini)
    return rows, global_numbers, weighted


def main():
    columns = ["polynomial", "method", "ring |z0|", "pixels", "converged (%)", "not converged (%)",
               "failed (%)", "mean evals (converged only)"]
    all_rows = []
    mism = 0
    for poly in POLYS:
        print(f"\n### {poly}", flush=True)
        rows, glob, weighted = ring_table(poly)
        all_rows += rows
        for name in glob:
            rs = [r for r in rows if r[1] == name]
            total = sum(r[3] for r in rs)
            w_conv = sum(r[3] * r[4] for r in rs) / total
            conv_px = [r[3] * r[4] / 100 for r in rs]
            w_evals = (sum(c * r[7] for c, r in zip(conv_px, rs) if r[7] != "nan") / sum(conv_px))
            print(f"- {poly} | {name}: ring pixels sum {total}; "
                  f"% converged by ring {[r[4] for r in rs]}; mean evals by ring {[r[7] for r in rs]}; "
                  f"ring-weighted: converged {w_conv:.4f}%, mean evals {w_evals:.3f}; "
                  f"global: converged {glob[name][0]:.4f}%, mean evals {glob[name][1]:.3f}", flush=True)
        mism += check_against_e2(poly, weighted)

    output = PROJECT_ROOT / "results" / "e12_radius_profile.csv"
    with open(output, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(columns)
        w.writerows(all_rows)
    print(f"\nTable saved to {output}; baseline checks that differ from e2: {mism}")

    # Figure: % converged by ring, one panel per polynomial.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True, sharey=True)
    for ax, poly in zip(axes.ravel(), POLYS):
        for name in dict.fromkeys(r[1] for r in all_rows):
            ys = [r[4] for r in all_rows if r[0] == poly and r[1] == name]
            ax.plot(range(len(RING_LABELS)), ys, marker="o", label=name)
        ax.set_xticks(range(len(RING_LABELS)))
        ax.set_xticklabels(["[0, 0.5)", "[0.5, 1)", "[1, 1.5)", "[1.5, 2)", "[2, 2√2]"], fontsize=8)
        ax.set_title(poly_latex(poly), fontsize=11)
        ax.set_xlabel("ring of $|z_0|$")
        ax.set_ylabel("converged (%)")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"Convergence by starting radius, window [−2, 2]², N = {N}, tol={TOL:g}, "
                 "max_iter: Newton 50, Halley 50, damped Newton 200, secant 100, Müller 100")
    fig.savefig(PROJECT_ROOT / "figures" / "e12_radius_profile.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("Figure saved to figures/e12_radius_profile.png")


def check_against_e2(poly, weighted):
    """The ring-weighted % converged and mean evaluations (exact, from the ring counts) must equal
    e2_summary.csv / e2_cost.csv once rounded the same way (4 and 3 decimals). Also the ring pixel counts
    must add up to N^2. Returns the number of methods that differ."""
    with open(PROJECT_ROOT / "results" / "e2_summary.csv", newline="", encoding="utf-8") as fh:
        s = {(r["polynomial"], r["method"]): r for r in csv.DictReader(fh)}
    with open(PROJECT_ROOT / "results" / "e2_cost.csv", newline="", encoding="utf-8") as fh:
        c = {(r["polynomial"], r["method"]): r for r in csv.DictReader(fh)}
    bad = 0
    for name, (n_px, w_conv, w_evals) in weighted.items():
        ok_px = n_px == N * N
        ok_conv = round(w_conv, 4) == float(s[(poly, name)]["converged (%)"])
        ok_evals = round(w_evals, 3) == float(c[(poly, name)]["mean evals"])
        bad += 0 if (ok_px and ok_conv and ok_evals) else 1
        print(f"check vs e2 {poly} | {name}: pixels {n_px} ({'ok' if ok_px else 'DIFFERENT'}), "
              f"converged {round(w_conv, 4)} vs {s[(poly, name)]['converged (%)']} "
              f"({'identical' if ok_conv else 'DIFFERENT'}), mean evals {round(w_evals, 3)} vs "
              f"{c[(poly, name)]['mean evals']} ({'identical' if ok_evals else 'DIFFERENT'})")
    return bad


if __name__ == "__main__":
    main()
