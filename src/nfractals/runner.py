import numpy as np

ACTIVO = 0
CONVERGIDO = 1
FALLO = 2
NO_CONVERGE = 3


def un_punto(fn):
    """Adapta un paso z -> z' (métodos de un solo punto) al contrato de estado."""
    return lambda s: (fn(s[0]),)


def run_basins(z0, step, f, roots, tol, max_iter, init=None):
    """Itera `step` sobre la malla z0 y clasifica cada punto.

    `step` recibe una tupla de arrays (el estado de los puntos activos) y
    devuelve la tupla del estado siguiente, con la misma longitud. El último
    elemento del estado es siempre la aproximación actual. `init` es el estado
    inicial (tupla de arrays con la forma de la malla); por defecto (z0,).
    Los métodos de un solo punto se adaptan con `un_punto`.

    Devuelve (raiz, iters, estado):
      raiz   -- índice de la raíz más cercana (argmin |z - roots|), -1 si no convergió
      iters  -- iteración (desde 1) en que convergió, 0 si no
      estado -- ACTIVO / CONVERGIDO / FALLO / NO_CONVERGE
    """
    if init is None:
        init = (z0,)
    S = tuple(np.array(a, copy=True) for a in init)
    shape = S[-1].shape
    if any(a.shape != shape for a in S):
        raise ValueError("todos los arrays del estado deben tener la forma de la malla")
    roots = np.asarray(roots)

    estado = np.full(shape, ACTIVO, dtype=np.int8)
    iters = np.zeros(shape, dtype=np.int32)
    raiz = np.full(shape, -1, dtype=np.int32)

    for it in range(1, max_iter + 1):
        activo = estado == ACTIVO
        if not activo.any():
            break

        nuevo = step(tuple(a[activo] for a in S))
        if len(nuevo) != len(S):
            raise ValueError("step debe devolver un estado de la misma longitud")
        for a, n in zip(S, nuevo):
            a[activo] = n
        z_nuevo = nuevo[-1]

        # Solo los puntos que seguían activos: fallo si el valor no es finito.
        finito = np.isfinite(z_nuevo)
        with np.errstate(all="ignore"):
            conv = finito & (np.abs(f(z_nuevo)) < tol)

        sub = np.full(z_nuevo.shape, ACTIVO, dtype=np.int8)
        sub[~finito] = FALLO
        sub[conv] = CONVERGIDO
        estado[activo] = sub

        # Índices planos de los que acaban de converger en esta iteración.
        idx = np.flatnonzero(activo)[conv]
        if idx.size:
            iters.flat[idx] = it
            dist = np.abs(z_nuevo[conv][:, None] - roots[None, :])
            raiz.flat[idx] = np.argmin(dist, axis=1)

    estado[estado == ACTIVO] = NO_CONVERGE
    return raiz, iters, estado
