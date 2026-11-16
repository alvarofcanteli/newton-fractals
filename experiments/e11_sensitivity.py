"""E11: sensitivity of the comparison to max_iter, N and the window.

Polynomials z^3 - 1, z^3 - 2z + 2, z^5 - 1 and z^6 - 1, the five methods, complex128, tol = 1e-10,
delta = 1e-3, criterion |z - r| < tol (same as E2).

Baseline setting = the one of E2: window [-2, 2]^2, N = 1000 and the default max_iter of each method
(Newton 50, Halley 50, damped Newton (h = 0.7) 200, secant 100, Muller 100). One parameter is varied
at a time:
  - max_iter in {20, 40, 100}: the SAME max_iter for the five methods (and the baseline, which uses
    the default of each method);
  - N in {250, 500, 1000} (window [-2, 2]^2);
  - window half-width in {1, 2, 4}, i.e. [-1, 1]^2, [-2, 2]^2, [-4, 4]^2 with N = 1000, so the pixel
    size changes (0.002, 0.004 and 0.008).
Means (iterations, evaluations) are over the converged points only.

Generates results/e11_sensitivity.csv (long format), results/e11_summary.csv (ranking of the methods
by mean evaluations in each setting) and results/e11_newton_z5_max_iter40.csv (Newton on z^5 - 1 with
max_iter = 40 in each window and each N).
"""
import csv
import sys
from pathlib import Path

import numpy as np

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
from nfractals.polynomials import POLYNOMIALS, make_polynomial, unique_roots  # noqa: E402
from nfractals.runner import CONVERGED, FAILED, NOT_CONVERGED, run_basins, single_point  # noqa: E402

POLYS = ["z^3 - 1", "z^3 - 2z + 2", "z^5 - 1", "z^6 - 1"]
TOL = 1e-10
H_DAMPED = 0.7
DELTA = 1e-3
BASE_N = 1000
BASE_HALF = 2.0
METHOD_NAMES = ["Newton", "Halley", f"Damped Newton (h={H_DAMPED})", "Secant", "Müller"]
DEFAULT_MAX_ITER = {"Newton": 50, "Halley": 50, f"Damped Newton (h={H_DAMPED})": 200,
                    "Secant": 100, "Müller": 100}

# (parameter, value, N, half-width, max_iter override or None)
SETTINGS = (
    [("max_iter", "baseline", BASE_N, BASE_HALF, None)]
    + [("max_iter", mi, BASE_N, BASE_HALF, mi) for mi in (20, 40, 100)]
    + [("N", n, n, BASE_HALF, None) for n in (250, 500, 1000)]
    + [("window", h, BASE_N, float(h), None) for h in (1, 2, 4)]
)

_poly_cache = {}


def polynomial(name):
    if name not in _poly_cache:
        f, df, d2f, roots_np = make_polynomial(POLYNOMIALS[name])
        _poly_cache[name] = (f, df, d2f, unique_roots(roots_np))
    return _poly_cache[name]


def grid(n, half):
    x = np.linspace(-half, half, n)
    return x[None, :] + 1j * x[:, None]


def method_spec(name, f, df, d2f, z0):
    """(step, evals per step, initial evals, initial state) of a method."""
    if name == "Newton":
        return single_point(lambda z: newton_step(z, f, df)), 2, 0, None
    if name == "Halley":
        return single_point(lambda z: halley_step(z, f, df, d2f)), 3, 0, None
    if name.startswith("Damped"):
        return single_point(lambda z: newton_step(z, f, df, h=H_DAMPED)), 2, 0, None
    if name == "Secant":
        return (lambda s: secant_step(s, f)), 1, 2, secant_init(z0, DELTA)
    return (lambda s: muller_step(s, f)), 1, 3, muller_init(z0, DELTA)


def run_cell(poly_name, method, n, half, max_iter=None):
    """Run `method` on `poly_name` at grid size n and window half-width `half`. `max_iter` None means the
    default of the method. Returns a dict with the percentages and the means (None if no point converges)."""
    f, df, d2f, roots = polynomial(poly_name)
    z0 = grid(n, half)
    step, cost, ini, init = method_spec(method, f, df, d2f, z0)
    mi = DEFAULT_MAX_ITER[method] if max_iter is None else max_iter
    _, iters, status = run_basins(z0, step, roots, TOL, mi, init=init)
    conv = status == CONVERGED
    it_mean = float(iters[conv].mean()) if conv.any() else float("nan")
    return {
        "max_iter": mi,
        "converged": 100 * float(np.mean(conv)),
        "not_converged": 100 * float(np.mean(status == NOT_CONVERGED)),
        "failed": 100 * float(np.mean(status == FAILED)),
        "mean_iter": it_mean,
        "mean_evals": it_mean * cost + ini,
    }


def ranking(means):
    """Methods ordered by mean evaluations (NaN last, ties by the order of METHOD_NAMES)."""
    key = lambda m: (np.inf if np.isnan(means[m]) else means[m], METHOD_NAMES.index(m))
    return sorted(METHOD_NAMES, key=key)


def main():
    long_columns = ["polynomial", "method", "parameter", "value", "max_iter", "N", "half_width",
                    "converged (%)", "not converged (%)", "failed (%)", "mean iter", "mean evals"]
    rows = []
    cache = {}  # (poly, method, N, half, effective max_iter) -> result
    means = {}  # (poly, parameter, value) -> {method: mean evals}

    for poly in POLYS:
        print(f"\n### {poly}", flush=True)
        for parameter, value, n, half, mi in SETTINGS:
            for method in METHOD_NAMES:
                eff = DEFAULT_MAX_ITER[method] if mi is None else mi
                key = (poly, method, n, half, eff)
                if key not in cache:
                    cache[key] = run_cell(poly, method, n, half, mi)
                r = cache[key]
                rows.append([poly, method, parameter, value, r["max_iter"], n, half,
                             round(r["converged"], 4), round(r["not_converged"], 4),
                             round(r["failed"], 4), round(r["mean_iter"], 3),
                             round(r["mean_evals"], 3)])
                means.setdefault((poly, parameter, value), {})[method] = round(r["mean_evals"], 3)
                print(f"- {poly} | {method} | {parameter}={value}: converged {r['converged']:.4f}%, "
                      f"not converged {r['not_converged']:.4f}%, failed {r['failed']:.4f}%, "
                      f"mean iter {r['mean_iter']:.3f}, mean evals {r['mean_evals']:.3f}", flush=True)

    # Baseline check against the earlier CSVs.
    mism = check_baseline(rows)

    # Ranking summary.
    summary = []
    for poly in POLYS:
        base_rank = ranking(means[(poly, "max_iter", "baseline")])
        for parameter, value, *_ in SETTINGS:
            rk = ranking(means[(poly, parameter, value)])
            moved = [f"{m}: {base_rank.index(m) + 1} -> {rk.index(m) + 1}"
                     for m in METHOD_NAMES if rk.index(m) != base_rank.index(m)]
            summary.append([poly, parameter, value, " < ".join(rk), str(rk == base_rank).lower(),
                            "; ".join(moved)])
            print(f"{poly} | {parameter}={value}: ranking {' < '.join(rk)}; "
                  f"same as baseline: {rk == base_rank}; changes: {'; '.join(moved) or 'none'}")

    # Newton on z^5 - 1 with max_iter = 40: each window and each N.
    q_rows = []
    for parameter, value, n, half in ([("window", h, BASE_N, float(h)) for h in (1, 2, 4)]
                                      + [("N", n, n, BASE_HALF) for n in (250, 500, 1000)]):
        r = cache.get(("z^5 - 1", "Newton", n, half, 40)) or run_cell("z^5 - 1", "Newton", n, half, 40)
        q_rows.append(["z^5 - 1", "Newton", 40, parameter, value, n, half, round(r["converged"], 4),
                       round(r["not_converged"], 4), round(r["failed"], 4)])
        print(f"Newton z^5-1 max_iter=40 {parameter}={value}: not converged {r['not_converged']:.4f}%, "
              f"failed {r['failed']:.4f}%", flush=True)

    write("e11_sensitivity.csv", long_columns, rows)
    write("e11_summary.csv", ["polynomial", "parameter", "value", "ranking by mean evaluations",
                              "same as baseline ranking", "methods that change position"], summary)
    write("e11_newton_z5_max_iter40.csv",
          ["polynomial", "method", "max_iter", "parameter", "value", "N", "half_width",
           "converged (%)", "not converged (%)", "failed (%)"], q_rows)
    print(f"\nBaseline rows that differ from the earlier CSVs: {mism}")


def write(name, columns, rows):
    output = PROJECT_ROOT / "results" / name
    with open(output, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(columns)
        w.writerows(rows)
    print(f"Table saved to {output}")


def read(name):
    with open(PROJECT_ROOT / "results" / name, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def check_baseline(rows):
    """Compare the baseline rows (parameter max_iter = baseline) with e2_cost.csv / e2_summary.csv
    (z^3 - 1, z^3 - 2z + 2, z^6 - 1) and with e5_polynomials.csv (z^5 - 1). Returns the number of
    mismatches (and prints every comparison)."""
    e2 = {(r["polynomial"], r["method"]): r for r in read("e2_cost.csv")}
    e2s = {(r["polynomial"], r["method"]): r for r in read("e2_summary.csv")}
    e5 = {(r["polynomial"], r["method"]): r for r in read("e5_polynomials.csv")}
    mism = 0
    for r in rows:
        poly, method, parameter, value = r[0], r[1], r[2], r[3]
        if parameter != "max_iter" or value != "baseline":
            continue
        mine = (r[7], r[8], r[9], r[10], r[11])  # converged, not conv., failed, mean iter, mean evals
        if poly in {p for p, _ in e2}:
            a, b = e2[(poly, method)], e2s[(poly, method)]
            ref = (b["converged (%)"], b["not converged (%)"], b["failed (%)"], a["mean iter"], a["mean evals"])
            src = "e2"
        else:
            a = e5[(poly, method)]
            ref = (a["converged (%)"], a["not converged (%)"], a["failed (%)"], a["mean iter"], a["mean evals"])
            src = "e5"
        ok = all(float(x) == float(y) for x, y in zip(mine, ref))
        mism += 0 if ok else 1
        print(f"baseline check {poly} | {method}: vs {src}: {'identical' if ok else 'DIFFERENT'} "
              f"(mine {mine}; reference {ref})")
    return mism


if __name__ == "__main__":
    main()
