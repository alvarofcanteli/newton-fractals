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

from nfractals.methods import (  # noqa: E402
    halley_step,
    muller_init,
    muller_step,
    newton_step,
    secant_init,
    secant_step,
)
from nfractals.polynomials import POLYNOMIALS, make_polynomial, unique_roots  # noqa: E402
from nfractals.runner import CONVERGED, run_basins, single_point  # noqa: E402

N = 1000
H_DAMPED = 0.7
DELTA = 1e-3
BOUNDARY_RADIUS = 3  # pixels, Chebyshev distance
EPS = {np.complex64: float(np.finfo(np.float32).eps), np.complex128: float(np.finfo(np.float64).eps)}
TOL = {np.complex64: 1e-5, np.complex128: 1e-10}

# Grid: built once in complex128, rounded to complex64 and converted back to complex128,
# so that both runs start from exactly the same points.
x = np.linspace(-2, 2, N)
z0_c128 = x[None, :] + 1j * x[:, None]
z0_c64 = z0_c128.astype(np.complex64)
z0_double = z0_c64.astype(np.complex128)
grids = {np.complex64: z0_c64, np.complex128: z0_double}

print("Tolerances and machine epsilon:")
for dt in (np.complex64, np.complex128):
    print(f"  {np.dtype(dt).name}: eps = {EPS[dt]:.2e}, tol = {TOL[dt]:.0e} = {TOL[dt] / EPS[dt]:.1e} eps. "
          "Evaluating f near a simple root has a noise floor of a few eps; "
          "tol is well above that floor (so that rounding noise is not reported as 'not converged') "
          "and well below the scale of f on the grid (O(1)).")

coef = POLYNOMIALS["z^3 - 1"]
systems = {}
for dt in (np.complex64, np.complex128):
    f, df, d2f, r = make_polynomial(coef, dtype=dt)
    systems[dt] = (f, df, d2f, unique_roots(r))


def methods_for(dt):
    f, df, d2f, _ = systems[dt]
    z0 = grids[dt]
    # (name, step, max_iter, initial state)
    return [
        ("Newton", single_point(lambda z: newton_step(z, f, df)), 50, None),
        ("Halley", single_point(lambda z: halley_step(z, f, df, d2f)), 50, None),
        (f"Damped Newton (h={H_DAMPED})",
         single_point(lambda z: newton_step(z, f, df, h=H_DAMPED)), 200, None),
        ("Secant", lambda s: secant_step(s, f), 100, secant_init(z0, DELTA)),
        ("Müller", lambda s: muller_step(s, f), 100, muller_init(z0, DELTA)),
    ]


def near_other_root(root, ok, radius):
    """Converged pixels within Chebyshev distance <= radius of some converged pixel
    with a different root."""
    h, w = root.shape
    near = np.zeros(root.shape, dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            a = (slice(max(0, dy), h + min(0, dy)), slice(max(0, dx), w + min(0, dx)))
            b = (slice(max(0, -dy), h + min(0, -dy)), slice(max(0, -dx), w + min(0, -dx)))
            near[a] |= ok[a] & ok[b] & (root[a] != root[b])
    return near


columns = [
    "method", "max_iter", "tol complex64", "tol complex128",
    "root differs (% of all)", "root differs (% of those converged in both)",
    "converged only in complex64 (%)", "converged only in complex128 (%)",
    "mean iter complex64", "mean iter complex128",
    "near boundary: % of all pixels",
    "near boundary: % of those that differ in root",
    "near boundary: % of those that differ in anything",
]
rows = []
res = {dt: [run_basins(grids[dt], step, systems[dt][3], TOL[dt], mi, init=init)
            for _, step, mi, init in methods_for(dt)]
       for dt in (np.complex64, np.complex128)}

print("\nResults (z^3 - 1, N = 1000, same starting points):")
for i, (name, _, max_iter, _) in enumerate(methods_for(np.complex128)):
    root64, it64, st64 = res[np.complex64][i]
    root128, it128, st128 = res[np.complex128][i]
    c64, c128 = st64 == CONVERGED, st128 == CONVERGED
    both = c64 & c128
    diff_root = both & (root64 != root128)
    only64, only128 = c64 & ~c128, c128 & ~c64
    diff_any = diff_root | only64 | only128

    boundary = near_other_root(root128, c128, BOUNDARY_RADIUS)
    p_bd_all = 100 * boundary.mean()
    p_bd_root = 100 * boundary[diff_root].mean() if diff_root.any() else float("nan")
    p_bd_any = 100 * boundary[diff_any].mean() if diff_any.any() else float("nan")

    p_diff_all = 100 * diff_root.mean()
    p_diff_both = 100 * diff_root.sum() / both.sum()
    i64, i128 = float(it64[c64].mean()), float(it128[c128].mean())
    rows.append([name, max_iter, TOL[np.complex64], TOL[np.complex128],
                 round(p_diff_all, 5), round(p_diff_both, 5),
                 round(100 * only64.mean(), 5), round(100 * only128.mean(), 5),
                 round(i64, 3), round(i128, 3), round(p_bd_all, 4),
                 round(p_bd_root, 4), round(p_bd_any, 4)])
    print(f"- {name} (max_iter={max_iter}): root differs in {p_diff_all:.4f}% of the pixels "
          f"({diff_root.sum()} pixels; {p_diff_both:.4f}% of those converged in both); "
          f"converged only in complex64 {100 * only64.mean():.4f}% ({only64.sum()}), "
          f"only in complex128 {100 * only128.mean():.4f}% ({only128.sum()}); "
          f"mean iter complex64 {i64:.3f}, complex128 {i128:.3f}; "
          f"near the boundary (≤{BOUNDARY_RADIUS} px): {p_bd_root:.2f}% of those that differ in root, "
          f"{p_bd_any:.2f}% of those that differ in anything, versus {p_bd_all:.2f}% of all pixels",
          flush=True)

    if name == "Newton":
        pixel_map = np.zeros(root128.shape, dtype=np.uint8)  # 0 background, 1 near boundary,
        pixel_map[boundary] = 1                              # 2 different root, 3 only in one
        pixel_map[only64 | only128] = 3
        pixel_map[diff_root] = 2
        fig, ax = plt.subplots(figsize=(9, 9), constrained_layout=True)
        ax.imshow(pixel_map, origin="lower", interpolation="nearest",
                  cmap=ListedColormap(["white", "#dddddd", "#d62728", "#1f77b4"]), vmin=0, vmax=3)
        ax.set_axis_off()
        ax.set_title(f"Newton, z^3 - 1: complex64 vs complex128 ({diff_root.sum()} pixels with a different root, "
                     f"{(only64 | only128).sum()} converged in only one)\n"
                     f"grey: within {BOUNDARY_RADIUS} px of another root; red: different root; "
                     "blue: converges in only one",
                     fontsize=10)
        output_fig = PROJECT_ROOT / "figures" / "e7_diff_newton.png"
        fig.savefig(output_fig, dpi=110, bbox_inches="tight")
        plt.close(fig)

output = PROJECT_ROOT / "results" / "e7_precision.csv"
with open(output, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(columns)
    w.writerows(rows)
print(f"\nTable saved to {output}")
