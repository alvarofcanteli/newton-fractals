import numpy as np


def newton_step(z, f, df, h=1.0):
    """One Newton step: z - h * f(z) / df(z). With h < 1 it is damped Newton.

    If z is complex64 or complex128 its dtype is kept; any other type
    (integer or real) is converted to complex128. Where df(z) = 0 the result
    is inf or NaN, without warnings.
    """
    z = np.asarray(z)
    if z.dtype not in (np.complex64, np.complex128):
        z = z.astype(np.complex128)
    with np.errstate(all="ignore"):
        out = z - h * f(z) / df(z)
    return out.astype(z.dtype, copy=False)


def halley_step(z, f, df, d2f):
    """One Halley step: z - 2 f df / (2 df^2 - f d2f).

    Same dtype handling and division-by-zero behaviour as newton_step.
    """
    z = np.asarray(z)
    if z.dtype not in (np.complex64, np.complex128):
        z = z.astype(np.complex128)
    with np.errstate(all="ignore"):
        fz = f(z)
        dfz = df(z)
        out = z - 2 * fz * dfz / (2 * dfz**2 - fz * d2f(z))
    return out.astype(z.dtype, copy=False)


def _to_complex(z):
    z = np.asarray(z)
    if z.dtype not in (np.complex64, np.complex128):
        z = z.astype(np.complex128)
    return z


def secant_init(z0, delta=1e-3):
    """Initial state of the secant method: (z0 + delta, z0) = (z_prev, z)."""
    z0 = _to_complex(z0)
    return (z0 + z0.dtype.type(delta), z0)


def secant_step(state, f):
    """One secant step: from (z_prev, z) to (z, z_new), with
    z_new = z - f(z) (z - z_prev) / (f(z) - f(z_prev)).

    Same dtype handling as newton_step. Where f(z) == f(z_prev) the result is
    inf or NaN, without warnings.
    """
    z_prev, z = (_to_complex(a) for a in state)
    with np.errstate(all="ignore"):
        fz = f(z)
        out = z - fz * (z - z_prev) / (fz - f(z_prev))
    return (z, out.astype(z.dtype, copy=False))


def muller_init(z0, delta=1e-3):
    """Initial state of Muller's method: (z0, z0 + delta, z0 + 2 delta) = (z_{k-2}, z_{k-1}, z_k)."""
    z0 = _to_complex(z0)
    d = z0.dtype.type(delta)
    return (z0, z0 + d, z0 + 2 * d)


def muller_step(state, f):
    """One Muller step: from (z_{k-2}, z_{k-1}, z_k) to (z_{k-1}, z_k, z_{k+1}).

    Parabola through the three points; z_{k+1} = z_k - 2c / (b ± sqrt(b^2 - 4ac)),
    with the sign that gives the denominator of larger modulus. Same dtype
    handling and division-by-zero behaviour as newton_step (inf or NaN, without
    warnings).
    """
    z2, z1, z0 = (_to_complex(a) for a in state)
    with np.errstate(all="ignore"):
        f2, f1, c = f(z2), f(z1), f(z0)
        h1 = z1 - z2
        h2 = z0 - z1
        d1 = (f1 - f2) / h1
        d2 = (c - f1) / h2
        a = (d2 - d1) / (h2 + h1)
        b = a * h2 + d2
        sq = np.sqrt(b * b - 4 * a * c)
        den_plus = b + sq
        den_minus = b - sq
        den = np.where(np.abs(den_plus) >= np.abs(den_minus), den_plus, den_minus)
        out = z0 - 2 * c / den
    return (z1, z0, out.astype(z0.dtype, copy=False))
