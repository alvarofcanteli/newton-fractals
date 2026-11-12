import csv
import sys
from pathlib import Path

import mpmath as mp

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

H_DAMPED = mp.mpf("0.7")
TOL = mp.mpf(10) ** -250


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
        # (name, step, initial state, linear?)
        ("Newton (z^3-1)", lambda s: newton_scalar(s[0], f, df), (z,), False),
        ("Halley (z^3-1)", lambda s: halley_scalar(s[0], f, df, d2f), (z,), False),
        ("Secant (z^3-1)", lambda s: secant_scalar(s, f), (1 + d * mp.mpf("1.2"), z), False),
        ("Müller (z^3-1)", lambda s: muller_scalar(s, f),
         (1 + d * mp.mpf("1.4"), 1 + d * mp.mpf("1.2"), z), False),
        ("Damped Newton h=0.7 (z^3-1)",
         lambda s: newton_scalar(s[0], f, df, h=H_DAMPED), (z,), True),
        ("Newton at double root ((z-1)^2 (z+1))", lambda s: newton_scalar(s[0], g, dg), (z,), True),
    ]
    rows, summary = [], []
    for name, step, state, linear in cases:
        es = errors(step, state, 1, tol=TOL)
        ps = orders(es)
        rs = ratios(es)
        for k, e in enumerate(es):
            p = ps.get(k)
            r = rs.get(k)
            rows.append([
                name, k, mp.nstr(e, 15),
                "" if p is None else mp.nstr(p, 10),
                "" if r is None else mp.nstr(r, 10),
            ])
        valid = [(k, p) for k, p in ps.items() if p is not None]
        last = [p for _, p in valid[-3:]]
        mean = sum(last) / len(last) if last else None
        summary.append((name, es, valid, rs, mean, len(last), linear))
    return rows, summary


rows, summary = run()

for name, es, valid, rs, mean, n_last, linear in summary:
    print(f"\n== {name} ==")
    print(f"errors: {len(es)} (e_0 = {mp.nstr(es[0], 3)}, e_final = {mp.nstr(es[-1], 3)})")
    print("p_k (k: value):", "  ".join(f"{k}: {mp.nstr(p, 6)}" for k, p in valid) or "none")
    if linear:
        rv = [(k, r) for k, r in rs.items() if r is not None]
        print("e_{k+1}/e_k (k: value):", "  ".join(f"{k}: {mp.nstr(r, 6)}" for k, r in rv))
        print(f"mean of the last {min(3, len(rv))} ratios: "
              f"{mp.nstr(sum(r for _, r in rv[-3:]) / len(rv[-3:]), 8)}")
    if mean is not None:
        print(f"mean of the last {n_last} p_k: {mp.nstr(mean, 8)}")

output = PROJECT_ROOT / "results" / "e4_order.csv"
with open(output, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["method", "k", "e_k", "p_k", "e_{k+1}/e_k"])
    w.writerows(rows)
print(f"\nData saved to {output}")
