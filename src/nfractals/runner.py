import numpy as np

ACTIVE = 0
CONVERGED = 1
FAILED = 2
NOT_CONVERGED = 3


def single_point(fn):
    """Adapt a step z -> z' (single-point methods) to the state contract."""
    return lambda s: (fn(s[0]),)


def run_basins(z0, step, f, roots, tol, max_iter, init=None):
    """Iterate `step` over the grid z0 and classify every point.

    `step` receives a tuple of arrays (the state of the active points) and
    returns the tuple of the next state, with the same length. The last
    element of the state is always the current approximation. `init` is the
    initial state (tuple of arrays with the grid shape); by default (z0,).
    Single-point methods are adapted with `single_point`.

    Returns (root, iters, status):
      root   -- index of the nearest root (argmin |z - roots|), -1 if it did not converge
      iters  -- iteration (counting from 1) at which it converged, 0 if it did not
      status -- ACTIVE / CONVERGED / FAILED / NOT_CONVERGED
    """
    if init is None:
        init = (z0,)
    state = tuple(np.array(a, copy=True) for a in init)
    shape = state[-1].shape
    if any(a.shape != shape for a in state):
        raise ValueError("all arrays of the state must have the grid shape")
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
        with np.errstate(all="ignore"):
            conv = finite & (np.abs(f(z_new)) < tol)

        sub = np.full(z_new.shape, ACTIVE, dtype=np.int8)
        sub[~finite] = FAILED
        sub[conv] = CONVERGED
        status[active] = sub

        # Flat indices of the points that have just converged in this iteration.
        idx = np.flatnonzero(active)[conv]
        if idx.size:
            iters.flat[idx] = it
            dist = np.abs(z_new[conv][:, None] - roots[None, :])
            root.flat[idx] = np.argmin(dist, axis=1)

    status[status == ACTIVE] = NOT_CONVERGED
    return root, iters, status
