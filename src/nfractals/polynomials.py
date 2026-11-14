import numpy as np


def make_polynomial(coeffs, dtype=np.complex128):
    """Return (f, df, d2f, roots). The functions compute in `dtype`;
    the reference roots are always in double precision."""
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
    """Coefficients of z^n - 1 (from highest to lowest degree)."""
    c = [0] * (n + 1)
    c[0] = 1
    c[-1] = -1
    return c


def random_poly(seed=2026, degree=5, r_min=0.3, r_max=1.5):
    """Polynomial with random roots: moduli uniform in [r_min, r_max), angles uniform in [0, 2*pi),
    drawn in this order from np.random.default_rng(seed). Returns (coefficients, generating roots);
    the coefficients are complex, monic, from highest to lowest degree (np.poly)."""
    rng = np.random.default_rng(seed)
    moduli = rng.uniform(r_min, r_max, degree)
    angles = rng.uniform(0, 2 * np.pi, degree)
    roots = moduli * np.exp(1j * angles)
    return np.poly(roots), roots


def unique_roots(roots, tol=1e-6):
    """Merge into a single root (by averaging) those closer than `tol` to each other.

    Useful with multiple roots: np.roots returns several almost identical ones and
    the argmin would split their basin. The roots come out in the order of the first
    appearance of each group; group membership is transitive (chains of roots closer
    than tol).
    """
    roots = np.asarray(roots)
    n = len(roots)
    group = -np.ones(n, dtype=int)
    n_groups = 0
    for i in range(n):
        if group[i] >= 0:
            continue
        group[i] = n_groups
        stack = [i]
        while stack:
            a = stack.pop()
            for b in range(n):
                if group[b] < 0 and abs(roots[a] - roots[b]) < tol:
                    group[b] = n_groups
                    stack.append(b)
        n_groups += 1
    return np.array([roots[group == g].mean() for g in range(n_groups)])


POLYNOMIALS = {
    "z^3 - 1": [1, 0, 0, -1],
    "z^3 - 2z + 2": [1, 0, -2, 2],
    "(z-1)^2 (z+1)": [1, -1, -1, 1],
    "z^4 - 1": unity_poly(4),
    "z^5 - 1": unity_poly(5),
    "z^6 - 1": unity_poly(6),
    "z^8 - 1": unity_poly(8),
    "(z-1)(z+1)(z-1.01)": np.poly([1, -1, 1.01]).tolist(),
    "random degree 5": random_poly()[0],
}
