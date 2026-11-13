import numpy as np
from matplotlib.colors import hsv_to_rgb

from nfractals.runner import CONVERGED

# Minimum brightness of a converged point (pure black is reserved for points that do not converge).
MIN_BRIGHTNESS = 0.35

# Mathtext names of the polynomials of `POLYNOMIALS`, for figure titles and legends.
POLY_LATEX = {
    "z^3 - 1": r"$z^3-1$",
    "z^3 - 2z + 2": r"$z^3-2z+2$",
    "(z-1)^2 (z+1)": r"$(z-1)^2(z+1)$",
    "z^4 - 1": r"$z^4-1$",
    "z^5 - 1": r"$z^5-1$",
    "z^6 - 1": r"$z^6-1$",
}


def poly_latex(name):
    """Mathtext version of a polynomial name from `POLYNOMIALS` (the name itself if unknown)."""
    return POLY_LATEX.get(name, name)


def plot_basins(root, iters, status, n_roots, ax=None, iter_ref=None):
    """Return an RGB image (height, width, 3) of the basins.

    Hue: according to the root (evenly spaced on the colour wheel).
    Brightness: decreases with the log of the iterations, from 1 (1 iteration) to
    MIN_BRIGHTNESS (iter_ref iterations or more, clipped above); darker = more
    iterations. If `iter_ref` is not given, the 99th percentile of the iterations
    of the converged points of this map is used.
    Black: points whose status is not CONVERGED.
    If `ax` is given, the image is also drawn on it.
    """
    conv = status == CONVERGED

    hue = np.where(conv, root, 0) / n_roots
    brightness = np.ones(root.shape)
    if conv.any():
        it = iters[conv].astype(float)
        if iter_ref is None:
            iter_ref = np.percentile(it, 99)
        # Logarithmic scale: 1 iteration -> rel 0 (brightness 1); iter_ref or more -> rel 1.
        if iter_ref > 1:
            rel = np.clip(np.log(it) / np.log(iter_ref), 0.0, 1.0)
        else:
            rel = np.zeros_like(it)
        brightness[conv] = 1.0 - (1.0 - MIN_BRIGHTNESS) * rel

    hsv = np.stack([hue, np.ones(root.shape), brightness], axis=-1)
    img = hsv_to_rgb(hsv)
    img[~conv] = 0.0

    if ax is not None:
        ax.imshow(img, origin="lower")
        ax.set_axis_off()
    return img
