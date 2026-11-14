import numpy as np

ACTIVE = 0
CONVERGED = 1
FAILED = 2
NOT_CONVERGED = 3


def single_point(fn):
    """Adapt a step z -> z' (single-point methods) to the state contract."""
    return lambda s: (fn(s[0]),)


def run_basins(z0, step, roots, tol, max_iter, init=None, return_final=False):
    """Iterate `step` over the grid z0 and classify every point.

    `step` receives a tuple of arrays (the state of the active points) and
    returns the tuple of the next state, with the same length. The last
    element of the state is always the current approximation. `init` is the
    initial state (tuple of arrays with the grid shape); by default (z0,).
    Single-point methods are adapted with `single_point`.

    A point converges when its distance to the nearest root is below `tol`,
    min_r |z - r| < tol, where `roots` are the distinct roots of the polynomial
    (see `unique_roots`). A non-finite value is FAILED.

    `roots` can also be a callable `nearest(z) -> (ids, dist)` that receives a 1-D array of finite
    values and returns, for each one, the integer id of its nearest reference root and the distance
    to it (see `nfractals.functions`). Then `root` holds those ids (which may be negative, so use
    `status` to tell converged points apart). With an array of roots the behaviour is the one above.
    If `return_final` is True a fourth array is returned: the last approximation of every point.

    Returns (root, iters, status):
      root   -- index of the nearest root (the one that triggered convergence), -1 if it did not converge
      iters  -- iteration (counting from 1) at which it converged, 0 if it did not
      status -- ACTIVE / CONVERGED / FAILED / NOT_CONVERGED
      (and, if `return_final`, z_final -- the last approximation of every point)
    """
    if init is None:
        init = (z0,)
    state = tuple(np.array(a, copy=True) for a in init)
    shape = state[-1].shape
    if any(a.shape != shape for a in state):
        raise ValueError("all arrays of the state must have the grid shape")
    nearest_fn = roots if callable(roots) else None
    if nearest_fn is None:
        roots = np.asarray(roots)

    status = np.full(shape, ACTIVE, dtype=np.int8)
    iters = np.zeros(shape, dtype=np.int32)
    root = np.full(shape, -1, dtype=np.int32)

    for it in range(1, max_iter + 1):
        active = status == ACTIVE
        if not active.any():
            break

        new = step(tuple(a[active] for a in state))
        if len(new) != len(state):
            raise ValueError("step must return a state of the same length")
        for a, n in zip(state, new):
            a[active] = n
        z_new = new[-1]

        # Only the points that were still active: failure if the value is not finite.
        finite = np.isfinite(z_new)
        nearest = np.zeros(z_new.shape, dtype=np.int32)
        dist_min = np.full(z_new.shape, np.inf)
        if finite.any():
            if nearest_fn is not None:
                nearest[finite], dist_min[finite] = nearest_fn(z_new[finite])
            else:
                dist = np.abs(z_new[finite][:, None] - roots[None, :])
                nearest[finite] = np.argmin(dist, axis=1)
                dist_min[finite] = dist[np.arange(dist.shape[0]), nearest[finite]]
        conv = finite & (dist_min < tol)

        sub = np.full(z_new.shape, ACTIVE, dtype=np.int8)
        sub[~finite] = FAILED
        sub[conv] = CONVERGED
        status[active] = sub

        # Flat indices of the points that have just converged in this iteration.
        idx = np.flatnonzero(active)[conv]
        if idx.size:
            iters.flat[idx] = it
            root.flat[idx] = nearest[conv]

    status[status == ACTIVE] = NOT_CONVERGED
    if return_final:
        return root, iters, status, state[-1]
    return root, iters, status
