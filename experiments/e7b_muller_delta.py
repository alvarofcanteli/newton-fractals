"""Müller in E7: effect of delta on the complex64 / complex128 difference and distance of
the pixels that differ to the boundaries.

Generates results/e7b_muller_delta.csv (one row per delta) and
figures/e7_diff_muller.png (delta = 1e-3).
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from nfractals.methods import muller_init, muller_step  # noqa: E402
from nfractals.polynomials import POLYNOMIALS, make_polynomial, unique_roots  # noqa: E402
from nfractals.runner import CONVERGED, run_basins  # noqa: E402

N = 1000
TOL = {np.complex64: 1e-5, np.complex128: 1e-10}
RADII = (3, 10, 20)  # boundary distances (in pixels) reported in the CSV

# Same grid as in E7: once in complex128, rounded to complex64 and converted back to complex128.
x = np.linspace(-2, 2, N)
z128 = x[None, :] + 1j * x[:, None]
z64 = z128.astype(np.complex64)
zd = z64.astype(np.complex128)


def dilate(mask, r):
    """Dilation with a square window of side 2r+1 (Chebyshev distance <= r)."""
    out = mask
    for axis in (0, 1):
        acc = out.copy()
        for s in range(1, r + 1):
            a = [slice(None), slice(None)]
            b = [slice(None), slice(None)]
            # Shift towards larger indices.
            a[axis], b[axis] = slice(s, None), slice(None, -s)
            acc[tuple(a)] |= out[tuple(b)]
            # Shift towards smaller indices.
            a[axis], b[axis] = slice(None, -s), slice(s, None)
            acc[tuple(a)] |= out[tuple(b)]
        out = acc
    return out


def near(root, ok, r):
    """Converged pixels within Chebyshev distance <= r of another converged pixel
    with a different root."""
    result = np.zeros(root.shape, bool)
    for k in np.unique(root[ok]):
        others = ok & (root != k)
        result |= (root == k) & ok & dilate(others, r)
    return result


def run(delta):
    out = {}
    for dt, z in ((np.complex64, z64), (np.complex128, zd)):
        f, _, _, rr = make_polynomial(POLYNOMIALS["z^3 - 1"], dtype=dt)
        out[dt] = run_basins(z, lambda s: muller_step(s, f), f, unique_roots(rr), TOL[dt], 100,
                             init=muller_init(z, delta))
    return out


columns = ["delta", "tol complex64", "tol complex128",
           "root differs (% of all)", "root differs (pixels)", "converged in only one (pixels)",
           "converged complex64 (%)", "converged complex128 (%)",
           "mean iter complex64", "mean iter complex128"]
for r in RADII:
    columns += [f"near boundary <={r} px: % of those that differ",
                f"near boundary <={r} px: % of all pixels"]
rows = []

for delta in (1e-1, 1e-2, 1e-3):
    o = run(delta)
    r64, i64, s64 = o[np.complex64]
    r128, i128, s128 = o[np.complex128]
    c64, c128 = s64 == CONVERGED, s128 == CONVERGED
    diff = c64 & c128 & (r64 != r128)
    only = c64 ^ c128
    print(f"delta={delta:g}: root differs {100 * diff.mean():.4f}% ({diff.sum()} px); "
          f"converged in only one: {only.sum()} px; conv. c64 {100 * c64.mean():.3f}%, "
          f"c128 {100 * c128.mean():.3f}%; mean iter c64 {i64[c64].mean():.3f}, "
          f"c128 {i128[c128].mean():.3f}")
    row = [delta, TOL[np.complex64], TOL[np.complex128],
           round(100 * diff.mean(), 4), int(diff.sum()), int(only.sum()),
           round(100 * c64.mean(), 3), round(100 * c128.mean(), 3),
           round(i64[c64].mean(), 3), round(i128[c128].mean(), 3)]
    for r in RADII:
        fr = near(r128, c128, r)
        pct = 100 * fr[diff].mean() if diff.any() else float("nan")
        print(f"   ≤{r} px: {pct:.2f}% of those that differ; {100 * fr.mean():.2f}% of all pixels")
        row += [round(pct, 2), round(100 * fr.mean(), 2)]
    rows.append(row)
    if delta == 1e-3:
        fr3 = near(r128, c128, 3)
        pixel_map = np.zeros(r128.shape, np.uint8)
        pixel_map[fr3] = 1
        pixel_map[only] = 3
        pixel_map[diff] = 2
        fig, ax = plt.subplots(figsize=(9, 9), constrained_layout=True)
        ax.imshow(pixel_map, origin="lower", interpolation="nearest",
                  cmap=ListedColormap(["white", "#dddddd", "#d62728", "#1f77b4"]), vmin=0, vmax=3)
        ax.set_axis_off()
        ax.set_title(f"Müller δ=1e-3, z^3 - 1: complex64 vs complex128 ({diff.sum()} pixels with a different root)\n"
                     "grey: within 3 px of another root; red: different root; blue: converges in only one",
                     fontsize=10)
        output = PROJECT_ROOT / "figures" / "e7_diff_muller.png"
        fig.savefig(output, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"Figure saved to {output}")

output = PROJECT_ROOT / "results" / "e7b_muller_delta.csv"
with open(output, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(columns)
    w.writerows(rows)
print(f"\nTable saved to {output}")
