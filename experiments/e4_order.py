import csv
import sys
from pathlib import Path

import mpmath as mp

sys.stdout.reconfigure(encoding="utf-8")

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]

from nfractals.convergence import (  # noqa: E402
    DPS,
    errores,
    halley_scalar,
    muller_scalar,
    newton_scalar,
    ordenes,
    razones,
    secant_scalar,
)

H_AMORTIGUADO = mp.mpf("0.7")
TOL = mp.mpf(10) ** -250


def corre():
    mp.mp.dps = DPS
    d = mp.mpf("0.05")
    # z^3 - 1, raíz 1
    f = lambda z: z**3 - 1
    df = lambda z: 3 * z**2
    d2f = lambda z: 6 * z
    # (z-1)^2 (z+1), raíz doble 1
    g = lambda z: (z - 1) ** 2 * (z + 1)
    dg = lambda z: 2 * (z - 1) * (z + 1) + (z - 1) ** 2

    z = 1 + d  # error inicial 0.05
    casos = [
        # (nombre, paso, estado inicial, lineal?)
        ("Newton (z^3-1)", lambda s: newton_scalar(s[0], f, df), (z,), False),
        ("Halley (z^3-1)", lambda s: halley_scalar(s[0], f, df, d2f), (z,), False),
        ("Secante (z^3-1)", lambda s: secant_scalar(s, f), (1 + d * mp.mpf("1.2"), z), False),
        ("Müller (z^3-1)", lambda s: muller_scalar(s, f),
         (1 + d * mp.mpf("1.4"), 1 + d * mp.mpf("1.2"), z), False),
        ("Newton amortiguado h=0.7 (z^3-1)",
         lambda s: newton_scalar(s[0], f, df, h=H_AMORTIGUADO), (z,), True),
        ("Newton en raíz doble ((z-1)^2 (z+1))", lambda s: newton_scalar(s[0], g, dg), (z,), True),
    ]
    filas, resumen = [], []
    for nombre, paso, estado, lineal in casos:
        es = errores(paso, estado, 1, tol=TOL)
        ps = ordenes(es)
        rs = razones(es)
        for k, e in enumerate(es):
            p = ps.get(k)
            r = rs.get(k)
            filas.append([
                nombre, k, mp.nstr(e, 15),
                "" if p is None else mp.nstr(p, 10),
                "" if r is None else mp.nstr(r, 10),
            ])
        validos = [(k, p) for k, p in ps.items() if p is not None]
        ult = [p for _, p in validos[-3:]]
        media = sum(ult) / len(ult) if ult else None
        resumen.append((nombre, es, validos, rs, media, len(ult), lineal))
    return filas, resumen


filas, resumen = corre()

for nombre, es, validos, rs, media, n_ult, lineal in resumen:
    print(f"\n== {nombre} ==")
    print(f"errores: {len(es)} (e_0 = {mp.nstr(es[0], 3)}, e_final = {mp.nstr(es[-1], 3)})")
    print("p_k (k: valor):", "  ".join(f"{k}: {mp.nstr(p, 6)}" for k, p in validos) or "ninguno")
    if lineal:
        rv = [(k, r) for k, r in rs.items() if r is not None]
        print("e_{k+1}/e_k (k: valor):", "  ".join(f"{k}: {mp.nstr(r, 6)}" for k, r in rv))
        print(f"media de las últimas {min(3, len(rv))} razones: "
              f"{mp.nstr(sum(r for _, r in rv[-3:]) / len(rv[-3:]), 8)}")
    if media is not None:
        print(f"media de los últimos {n_ult} p_k: {mp.nstr(media, 8)}")

salida = RAIZ_PROYECTO / "results" / "e4_order.csv"
with open(salida, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["metodo", "k", "e_k", "p_k", "e_{k+1}/e_k"])
    w.writerows(filas)
print(f"\nDatos guardados en {salida}")
