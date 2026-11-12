import numpy as np


def make_polynomial(coeffs, dtype=np.complex128):
    """Devuelve (f, df, d2f, roots). Las funciones calculan en `dtype`;
    las raíces de referencia siempre en doble precisión."""
    p64 = np.asarray(coeffs, dtype=np.complex128)
    roots = np.roots(p64)

    p = p64.astype(dtype)
    dp = np.polyder(p)
    d2p = np.polyder(dp)

    def f(z):
        return np.polyval(p, np.asarray(z, dtype=dtype))

    def df(z):
        return np.polyval(dp, np.asarray(z, dtype=dtype))

    def d2f(z):
        return np.polyval(d2p, np.asarray(z, dtype=dtype))

    return f, df, d2f, roots


def unity_poly(n):
    """Coeficientes de z^n - 1 (de mayor a menor grado)."""
    c = [0] * (n + 1)
    c[0] = 1
    c[-1] = -1
    return c


def unique_roots(roots, tol=1e-6):
    """Agrupa en una sola raíz (promediando) las que distan menos de `tol` entre sí.

    Útil con raíces múltiples: np.roots devuelve varias casi idénticas y el argmin
    partiría su cuenca. Las raíces quedan en el orden de la primera aparición de cada
    grupo; la pertenencia a un grupo es transitiva (cadenas de raíces a menos de tol).
    """
    roots = np.asarray(roots)
    n = len(roots)
    grupo = -np.ones(n, dtype=int)
    n_grupos = 0
    for i in range(n):
        if grupo[i] >= 0:
            continue
        grupo[i] = n_grupos
        pila = [i]
        while pila:
            a = pila.pop()
            for b in range(n):
                if grupo[b] < 0 and abs(roots[a] - roots[b]) < tol:
                    grupo[b] = n_grupos
                    pila.append(b)
        n_grupos += 1
    return np.array([roots[grupo == g].mean() for g in range(n_grupos)])


POLYNOMIALS = {
    "z^3 - 1": [1, 0, 0, -1],
    "z^3 - 2z + 2": [1, 0, -2, 2],
    "(z-1)^2 (z+1)": [1, -1, -1, 1],
    "z^4 - 1": unity_poly(4),
    "z^5 - 1": unity_poly(5),
    "z^6 - 1": unity_poly(6),
}
