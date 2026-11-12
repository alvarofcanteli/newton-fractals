"""Versiones escalares con mpmath de los métodos de methods.py, para medir órdenes de convergencia.

Mismas fórmulas que methods.py, pero sin NumPy: cada paso trabaja con la precisión
de mpmath vigente (el llamador fija mp.dps, p. ej. con mp.workdps(50)).
"""

import mpmath as mp

DPS = 300
# Los errores por debajo de este umbral están dominados por el redondeo de DPS dígitos
# (z ≈ 1 solo tiene precisión absoluta ~1e-DPS) y no se usan para estimar órdenes.
UMBRAL_FIABLE = mp.mpf(10) ** -290


def newton_scalar(z, f, df, h=1):
    """z - h f(z)/f'(z). Con h < 1 es Newton amortiguado."""
    return (z - h * f(z) / df(z),)


def halley_scalar(z, f, df, d2f):
    """z - 2 f f' / (2 f'^2 - f f'')."""
    fz = f(z)
    dfz = df(z)
    return (z - 2 * fz * dfz / (2 * dfz**2 - fz * d2f(z)),)


def secant_scalar(estado, f):
    """(z_ant, z) -> (z, z_nuevo)."""
    z_ant, z = estado
    fz = f(z)
    return (z, z - fz * (z - z_ant) / (fz - f(z_ant)))


def muller_scalar(estado, f):
    """(z_{k-2}, z_{k-1}, z_k) -> (z_{k-1}, z_k, z_{k+1}); denominador de mayor módulo."""
    z2, z1, z0 = estado
    f2, f1, c = f(z2), f(z1), f(z0)
    h1 = z1 - z2
    h2 = z0 - z1
    d1 = (f1 - f2) / h1
    d2 = (c - f1) / h2
    a = (d2 - d1) / (h2 + h1)
    b = a * h2 + d2
    raiz = mp.sqrt(b * b - 4 * a * c)
    den = max(b + raiz, b - raiz, key=abs)
    return (z1, z0, z0 - 2 * c / den)


def errores(paso, estado, raiz, tol=mp.mpf(10) ** -250, max_iter=2000):
    """Itera `paso` (tupla -> tupla, última = aproximación actual) y devuelve
    [e_0, e_1, ...] con e_k = |z_k - raiz|, hasta que e_k < tol (inclusive)."""
    es = [abs(estado[-1] - raiz)]
    for _ in range(max_iter):
        if es[-1] < tol:
            break
        try:
            estado = paso(estado)
        except ZeroDivisionError:
            break
        es.append(abs(estado[-1] - raiz))
    return es


def ordenes(es):
    """p_k = ln(e_{k+1}/e_k) / ln(e_k/e_{k-1}) para k = 1..len-2, o None si algún
    error de la terna es cero o menor que UMBRAL_FIABLE."""
    ps = {}
    for k in range(1, len(es) - 1):
        terna = es[k - 1 : k + 2]
        if any(e < UMBRAL_FIABLE for e in terna):
            ps[k] = None
        else:
            ps[k] = mp.log(es[k + 1] / es[k]) / mp.log(es[k] / es[k - 1])
    return ps


def razones(es):
    """r_k = e_{k+1}/e_k para k = 0..len-2 (None si e_k es cero o no fiable)."""
    return {
        k: (es[k + 1] / es[k] if es[k] >= UMBRAL_FIABLE and es[k + 1] >= UMBRAL_FIABLE else None)
        for k in range(len(es) - 1)
    }
