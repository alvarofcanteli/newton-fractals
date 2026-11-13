import numpy as np

from nfractals.runner import CONVERGED


def boundary_mask(root, status):
    """Boolean mask of the boundary pixels: converged pixels that have some converged neighbour
    (4-connectivity) with another root. Pairs in which some pixel does not converge do not count."""
    ok = status == CONVERGED
    mark = np.zeros(root.shape, dtype=bool)
    # Vertical neighbours.
    d = ok[1:, :] & ok[:-1, :] & (root[1:, :] != root[:-1, :])
    mark[1:, :] |= d
    mark[:-1, :] |= d
    # Horizontal neighbours.
    d = ok[:, 1:] & ok[:, :-1] & (root[:, 1:] != root[:, :-1])
    mark[:, 1:] |= d
    mark[:, :-1] |= d
    return mark


def boundary_fraction(root, status):
    """Fraction of grid pixels that are boundary pixels (see `boundary_mask`)."""
    return boundary_mask(root, status).mean()  # over all pixels of the grid
