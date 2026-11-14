"""Non-polynomial functions for E10 (an extension beyond the polynomial scope of the course guide).

Each `Function` bundles f, f', f'', the window of the grid and a `nearest` callable that assigns
every finite value its nearest reference root (integer id and distance), the interface accepted by
`run_basins`. `group_of` maps root ids to the groups used in tables and colours.
"""
from dataclasses import dataclass
from typing import Callable

import mpmath
import numpy as np


@dataclass
class Function:
    name: str
    latex: str
    f: Callable
    df: Callable
    d2f: Callable
    half_width: float  # window [-half_width, half_width]^2
    nearest: Callable  # z (1-D, finite) -> (ids, distance to the nearest reference root)
    group_labels: list
    group_of: Callable  # ids -> index of the group (0 .. len(group_labels) - 1)


def sin_nearest(z):
    """Nearest root of sin z: k*pi with k = round(Re z / pi), exactly. Returns (k, |z - k*pi|)."""
    k = np.round(z.real / np.pi)
    dist = np.abs(z - k * np.pi)
    return np.clip(k, -2**30, 2**30).astype(np.int32), dist


def lambert_roots(kmax=20, dps=50):
    """Roots of e^z - z: z_k = -W_k(-1) for |k| <= kmax (mpmath, `dps` digits), cast to complex128.
    Returns (ks, roots)."""
    ks = np.arange(-kmax, kmax + 1)
    with mpmath.workdps(dps):
        roots = np.array([complex(-mpmath.lambertw(-1, int(k))) for k in ks], dtype=np.complex128)
    return ks, roots


def make_exp_minus_z_nearest(ks, roots, chunk=100_000):
    """Nearest reference root of e^z - z among `roots` (ids are the Lambert branches `ks`)."""

    def nearest(z):
        ids = np.empty(z.shape, dtype=np.int32)
        dist = np.empty(z.shape)
        for a in range(0, z.size, chunk):
            d = np.abs(z[a:a + chunk, None] - roots[None, :])
            j = np.argmin(d, axis=1)
            ids[a:a + chunk] = ks[j]
            dist[a:a + chunk] = d[np.arange(j.size), j]
        return ids, dist

    return nearest


def make_functions():
    """The two functions of E10: sin z and e^z - z."""
    sin = Function(
        name="sin z",
        latex=r"$\sin z$",
        f=lambda z: np.sin(z),
        df=lambda z: np.cos(z),
        d2f=lambda z: -np.sin(z),
        half_width=5.0,
        nearest=sin_nearest,
        group_labels=["k=-2", "k=-1", "k=0", "k=1", "k=2", "other (|k|>=3)"],
        group_of=lambda ids: np.where(np.abs(ids) <= 2, ids + 2, 5),
    )
    ks, roots = lambert_roots()
    exp_minus_z = Function(
        name="e^z - z",
        latex=r"$e^z-z$",
        f=lambda z: np.exp(z) - z,
        df=lambda z: np.exp(z) - 1,
        d2f=lambda z: np.exp(z),
        half_width=3.0,
        nearest=make_exp_minus_z_nearest(ks, roots),
        group_labels=[f"{roots[ks == 0][0]:.4f}", f"{roots[ks == -1][0]:.4f}",
                      "other listed roots (|k|<=20)"],
        group_of=lambda ids: np.where(ids == 0, 0, np.where(ids == -1, 1, 2)),
    )
    exp_minus_z.roots = roots  # reference roots, for the tests and the CSV
    return [sin, exp_minus_z]
