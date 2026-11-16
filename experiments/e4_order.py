"""E4: order of convergence with mpmath at 300 digits.

z^3 - 1 (root 1) and the double root of (z-1)^2 (z+1) are the original cases. The simple-root cases
are also measured, from one hard-coded starting point each, for z^3 - 2z + 2, z^6 - 1 and the random
degree-5 polynomial (seed 2026); see NEW_CASES for why those points were chosen.

Generates results/e4_order.csv (the whole series, with a `polynomial` column) and
results/e4_summary.csv (one row per polynomial and method).
"""
import csv
import sys
from pathlib import Path

import mpmath as mp
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from nfractals.convergence import (  # noqa: E402
    DPS,
    errors,
    halley_scalar,
    muller_scalar,
    newton_scalar,
    orders,
    ratios,
    secant_scalar,
)
from nfractals.polynomials import POLYNOMIALS  # noqa: E402

H_DAMPED = mp.mpf("0.7")
TOL = mp.mpf(10) ** -250

# Simple-root cases for the new polynomials: (polynomial, hard-coded starting point z0).
# Why these points: each z0 lies at distance about 0.05 from ONE simple root (as the case of z^3 - 1,
# where the initial error is 0.05), so that Newton, Halley, damped Newton, secant and Muller all
# converge to that root from the very start and the asymptotic orders can be read from the series.
# z0 is complex because the roots of these polynomials are complex (z^3 - 2z + 2: 0.8846+0.5897i;
# z^6 - 1: exp(i pi / 3); random degree 5: 0.883778-0.599448i, the root farthest from the others).
# `check_start` verifies the choice: z0 is nearer to that root than to any other, the root is simple
# (f' != 0), the five methods end within TOL of it and no iterate leaves the half-distance to the
# nearest other root. The secant and Muller start from the same z0 and extra points further out on
# the line from the root through z0, as in the z^3 - 1 case: r + 1.2 (z0 - r) and r + 1.4 (z0 - r).
NEW_CASES = [
    ("z^3 - 2z + 2", mp.mpc("0.92", "0.62")),
    ("z^6 - 1", mp.mpc("0.54", "0.90")),
    ("random degree 5", mp.mpc("0.93", "-0.62")),
]


def mp_polynomial(name):
    """Exact mpmath functions for the polynomial `name` of POLYNOMIALS (the coefficients are the
    float64 ones used by the rest of the project, converted exactly)."""
    cs = [mp.mpc(complex(c)) for c in np.asarray(POLYNOMIALS[name], dtype=complex)]
    n = len(cs) - 1
    dcs = [c * (n - i) for i, c in enumerate(cs[:-1])]
    d2cs = [c * (n - 1 - i) for i, c in enumerate(dcs[:-1])]

    def horner(coefs):
        def fn(z):
            r = mp.mpc(0)
            for c in coefs:
                r = r * z + c
            return r
        return fn

    return horner(cs), horner(dcs), horner(d2cs), cs


def mp_roots(name, cs):
    """All roots of the polynomial at DPS + 30 digits: np.roots refined with Newton in mpmath."""
    f, df, _, _ = mp_polynomial(name)
    out = []
    with mp.workdps(DPS + 30):
        for r0 in np.roots(np.asarray(POLYNOMIALS[name], dtype=complex)):
            r = mp.mpc(complex(r0))
            for _ in range(12):
                r = r - f(r) / df(r)
            out.append(r)
    return out


def new_cases(name, z0):
    """(name, step, initial state, linear?) for the five methods on a new polynomial, the root and
    the other roots."""
    f, df, d2f, cs = mp_polynomial(name)
    roots = mp_roots(name, cs)
    r = min(roots, key=lambda q: abs(q - z0))
    others = [q for q in roots if q is not r]
    z = z0
    p12 = r + mp.mpf("1.2") * (z0 - r)
    p14 = r + mp.mpf("1.4") * (z0 - r)
    cases = [
        (f"Newton ({name})", lambda s: newton_scalar(s[0], f, df), (z,), False),
        (f"Halley ({name})", lambda s: halley_scalar(s[0], f, df, d2f), (z,), False),
        (f"Secant ({name})", lambda s: secant_scalar(s, f), (p12, z), False),
        (f"Müller ({name})", lambda s: muller_scalar(s, f), (p14, p12, z), False),
        (f"Damped Newton h=0.7 ({name})",
         lambda s: newton_scalar(s[0], f, df, h=H_DAMPED), (z,), True),
    ]
    return cases, r, others, df


def check_start(name, z0, cases, r, others, df, series):
    """Assertions that justify the hard-coded starting point (see NEW_CASES)."""
    assert all(abs(z0 - r) < abs(z0 - q) for q in others), f"{name}: z0 is not nearest to its root"
    assert abs(df(r)) > mp.mpf("0.1"), f"{name}: the root is not simple"
    half = min(abs(r - q) for q in others) / 2
    for (label, _, _, _), es in zip(cases, series):
        assert es[-1] < TOL, f"{label}: did not reach TOL"
        assert max(es) < half, f"{label}: an iterate went farther than half the distance to another root"


def run():
    mp.mp.dps = DPS
    d = mp.mpf("0.05")
    # z^3 - 1, root 1
    f = lambda z: z**3 - 1
    df = lambda z: 3 * z**2
    d2f = lambda z: 6 * z
    # (z-1)^2 (z+1), double root 1
    g = lambda z: (z - 1) ** 2 * (z + 1)
    dg = lambda z: 2 * (z - 1) * (z + 1) + (z - 1) ** 2

    z = 1 + d  # initial error 0.05
    cases = [
        # (polynomial, name, step, initial state, linear?, root, start)
        ("z^3 - 1", "Newton (z^3-1)", lambda s: newton_scalar(s[0], f, df), (z,), False),
        ("z^3 - 1", "Halley (z^3-1)", lambda s: halley_scalar(s[0], f, df, d2f), (z,), False),
        ("z^3 - 1", "Secant (z^3-1)", lambda s: secant_scalar(s, f), (1 + d * mp.mpf("1.2"), z), False),
        ("z^3 - 1", "Müller (z^3-1)", lambda s: muller_scalar(s, f),
         (1 + d * mp.mpf("1.4"), 1 + d * mp.mpf("1.2"), z), False),
        ("z^3 - 1", "Damped Newton h=0.7 (z^3-1)",
         lambda s: newton_scalar(s[0], f, df, h=H_DAMPED), (z,), True),
        ("(z-1)^2 (z+1)", "Newton at double root ((z-1)^2 (z+1))",
         lambda s: newton_scalar(s[0], g, dg), (z,), True),
    ]
    roots_of = {"z^3 - 1": mp.mpf(1), "(z-1)^2 (z+1)": mp.mpf(1)}
    starts = {"z^3 - 1": z, "(z-1)^2 (z+1)": z}
    for name, z0 in NEW_CASES:
        new, r, others, dfp = new_cases(name, z0)
        # Series first (needs the root), the assertions come after the loop below.
        for label, step, state, linear in new:
            cases.append((name, label, step, state, linear))
        roots_of[name], starts[name] = r, z0
        roots_of[(name, "check")] = (new, others, dfp)

    rows, summary = [], []
    series_of = {}
    for poly, name, step, state, linear in cases:
        root = roots_of[poly]
        es = errors(step, state, root, tol=TOL)
        series_of.setdefault(poly, []).append(es)
        ps = orders(es)
        rs = ratios(es)
        for k, e in enumerate(es):
            p = ps.get(k)
            r = rs.get(k)
            rows.append([
                poly, name, k, mp.nstr(e, 15),
                "" if p is None else mp.nstr(p, 10),
                "" if r is None else mp.nstr(r, 10),
            ])
        valid = [(k, p) for k, p in ps.items() if p is not None]
        last = [p for _, p in valid[-3:]]
        mean = sum(last) / len(last) if last else None
        summary.append((poly, name, es, valid, rs, mean, len(last), linear))

    for name, z0 in NEW_CASES:
        new, others, dfp = roots_of[(name, "check")]
        check_start(name, z0, new, roots_of[name], others, dfp, series_of[name])
    return rows, summary, roots_of, starts


def main():
    rows, summary, roots_of, starts = run()
    summary_rows = []
    for poly, name, es, valid, rs, mean, n_last, linear in summary:
        print(f"\n== {name} ==")
        print(f"errors: {len(es)} (e_0 = {mp.nstr(es[0], 3)}, e_final = {mp.nstr(es[-1], 3)})")
        print("p_k (k: value):", "  ".join(f"{k}: {mp.nstr(p, 6)}" for k, p in valid) or "none")
        ratio_mean = None
        if linear:
            rv = [(k, r) for k, r in rs.items() if r is not None]
            print("e_{k+1}/e_k (k: value):", "  ".join(f"{k}: {mp.nstr(r, 6)}" for k, r in rv))
            ratio_mean = sum(r for _, r in rv[-3:]) / len(rv[-3:])
            print(f"mean of the last {min(3, len(rv))} ratios: {mp.nstr(ratio_mean, 8)}")
        if mean is not None:
            print(f"mean of the last {n_last} p_k: {mp.nstr(mean, 8)}")
        z0 = starts[poly]
        summary_rows.append([
            poly, name, mp.nstr(roots_of[poly], 8), mp.nstr(z0, 8), mp.nstr(es[0], 6), len(es) - 1,
            mp.nstr(es[-1], 6), "" if mean is None else mp.nstr(mean, 8), "linear" if linear else "",
            "" if ratio_mean is None else mp.nstr(ratio_mean, 8),
        ])

    output = PROJECT_ROOT / "results" / "e4_order.csv"
    with open(output, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["polynomial", "method", "k", "e_k", "p_k", "e_{k+1}/e_k"])
        w.writerows(rows)
    print(f"\nData saved to {output}")

    output_summary = PROJECT_ROOT / "results" / "e4_summary.csv"
    with open(output_summary, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["polynomial", "method", "root", "start z0", "e_0", "iterations", "e_final",
                    "order (mean of the last 3 p_k)", "linear case",
                    "mean of the last 3 ratios e_{k+1}/e_k (linear cases)"])
        w.writerows(summary_rows)
    print(f"Summary saved to {output_summary}")


if __name__ == "__main__":
    main()
