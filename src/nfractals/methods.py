import numpy as np


def newton_step(z, f, df, h=1.0):
    """Un paso de Newton: z - h * f(z) / df(z). Con h < 1 es Newton amortiguado.

    Si z es complex64 o complex128 conserva su dtype; cualquier otro tipo
    (entero o real) se convierte a complex128. Donde df(z) = 0 deja inf o
    NaN, sin avisos.
    """
    z = np.asarray(z)
    if z.dtype not in (np.complex64, np.complex128):
        z = z.astype(np.complex128)
    with np.errstate(all="ignore"):
        out = z - h * f(z) / df(z)
    return out.astype(z.dtype, copy=False)


def halley_step(z, f, df, d2f):
    """Un paso de Halley: z - 2 f df / (2 df^2 - f d2f).

    Misma gestión de dtype y de divisiones por cero que newton_step.
    """
    z = np.asarray(z)
    if z.dtype not in (np.complex64, np.complex128):
        z = z.astype(np.complex128)
    with np.errstate(all="ignore"):
        fz = f(z)
        dfz = df(z)
        out = z - 2 * fz * dfz / (2 * dfz**2 - fz * d2f(z))
    return out.astype(z.dtype, copy=False)


def _a_complejo(z):
    z = np.asarray(z)
    if z.dtype not in (np.complex64, np.complex128):
        z = z.astype(np.complex128)
    return z


def secant_init(z0, delta=1e-3):
    """Estado inicial de la secante: (z0 + delta, z0) = (z_anterior, z)."""
    z0 = _a_complejo(z0)
    return (z0 + z0.dtype.type(delta), z0)


def secant_step(estado, f):
    """Un paso de la secante: de (z_anterior, z) a (z, z_nuevo), con
    z_nuevo = z - f(z) (z - z_anterior) / (f(z) - f(z_anterior)).

    Misma gestión de dtype que newton_step. Donde f(z) == f(z_anterior) deja
    inf o NaN, sin avisos.
    """
    z_ant, z = (_a_complejo(a) for a in estado)
    with np.errstate(all="ignore"):
        fz = f(z)
        out = z - fz * (z - z_ant) / (fz - f(z_ant))
    return (z, out.astype(z.dtype, copy=False))


def muller_init(z0, delta=1e-3):
    """Estado inicial de Müller: (z0, z0 + delta, z0 + 2 delta) = (z_{k-2}, z_{k-1}, z_k)."""
    z0 = _a_complejo(z0)
    d = z0.dtype.type(delta)
    return (z0, z0 + d, z0 + 2 * d)


def muller_step(estado, f):
    """Un paso de Müller: de (z_{k-2}, z_{k-1}, z_k) a (z_{k-1}, z_k, z_{k+1}).

    Parábola por los tres puntos; z_{k+1} = z_k - 2c / (b ± sqrt(b^2 - 4ac)),
    con el signo que da el denominador de mayor módulo. Misma gestión de dtype
    y de divisiones por cero que newton_step (inf o NaN, sin avisos).
    """
    z2, z1, z0 = (_a_complejo(a) for a in estado)
    with np.errstate(all="ignore"):
        f2, f1, c = f(z2), f(z1), f(z0)
        h1 = z1 - z2
        h2 = z0 - z1
        d1 = (f1 - f2) / h1
        d2 = (c - f1) / h2
        a = (d2 - d1) / (h2 + h1)
        b = a * h2 + d2
        raiz = np.sqrt(b * b - 4 * a * c)
        den_mas = b + raiz
        den_menos = b - raiz
        den = np.where(np.abs(den_mas) >= np.abs(den_menos), den_mas, den_menos)
        out = z0 - 2 * c / den
    return (z1, z0, out.astype(z0.dtype, copy=False))
