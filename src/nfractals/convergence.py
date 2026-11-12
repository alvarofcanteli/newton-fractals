"""Scalar mpmath versions of the methods in methods.py, to measure convergence orders.

Same formulas as methods.py, but without NumPy: each step works with the current
mpmath precision (the caller sets mp.dps, e.g. with mp.workdps(50)).
"""

import mpmath as mp

DPS = 300
# Errors below this threshold are dominated by the rounding of DPS digits
# (z ≈ 1 only has absolute precision ~1e-DPS) and are not used to estimate orders.
RELIABLE_THRESHOLD = mp.mpf(10) ** -290


def newton_scalar(z, f, df, h=1):
    """z - h f(z)/f'(z). With h < 1 it is damped Newton."""
    return (z - h * f(z) / df(z),)


def halley_scalar(z, f, df, d2f):
    """z - 2 f f' / (2 f'^2 - f f'')."""
    fz = f(z)
    dfz = df(z)
    return (z - 2 * fz * dfz / (2 * dfz**2 - fz * d2f(z)),)


def secant_scalar(state, f):
    """(z_prev, z) -> (z, z_new)."""
    z_prev, z = state
    fz = f(z)
    return (z, z - fz * (z - z_prev) / (fz - f(z_prev)))


def muller_scalar(state, f):
    """(z_{k-2}, z_{k-1}, z_k) -> (z_{k-1}, z_k, z_{k+1}); denominator of larger modulus."""
    z2, z1, z0 = state
    f2, f1, c = f(z2), f(z1), f(z0)
    h1 = z1 - z2
    h2 = z0 - z1
    d1 = (f1 - f2) / h1
    d2 = (c - f1) / h2
    a = (d2 - d1) / (h2 + h1)
    b = a * h2 + d2
    sq = mp.sqrt(b * b - 4 * a * c)
    den = max(b + sq, b - sq, key=abs)
    return (z1, z0, z0 - 2 * c / den)


def errors(step, state, root, tol=mp.mpf(10) ** -250, max_iter=2000):
    """Iterate `step` (tuple -> tuple, last = current approximation) and return
    [e_0, e_1, ...] with e_k = |z_k - root|, until e_k < tol (inclusive)."""
    es = [abs(state[-1] - root)]
    for _ in range(max_iter):
        if es[-1] < tol:
            break
        try:
            state = step(state)
        except ZeroDivisionError:
            break
        es.append(abs(state[-1] - root))
    return es


def orders(es):
    """p_k = ln(e_{k+1}/e_k) / ln(e_k/e_{k-1}) for k = 1..len-2, or None if any
    error of the triple is zero or below RELIABLE_THRESHOLD."""
    ps = {}
    for k in range(1, len(es) - 1):
        triple = es[k - 1 : k + 2]
        if any(e < RELIABLE_THRESHOLD for e in triple):
            ps[k] = None
        else:
            ps[k] = mp.log(es[k + 1] / es[k]) / mp.log(es[k] / es[k - 1])
    return ps


def ratios(es):
    """r_k = e_{k+1}/e_k for k = 0..len-2 (None if e_k is zero or unreliable)."""
    return {
        k: (es[k + 1] / es[k] if es[k] >= RELIABLE_THRESHOLD and es[k + 1] >= RELIABLE_THRESHOLD else None)
        for k in range(len(es) - 1)
    }
