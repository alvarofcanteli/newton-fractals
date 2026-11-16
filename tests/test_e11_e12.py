"""Tests of E11 (sensitivity) and E12 (radius profile). The functions are tested on small grids; the CSVs
produced by the full runs are checked for layout and for the baseline numbers of E2."""
import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # the scripts have a main guard: importing does not run them
    return mod


@pytest.fixture(scope="module")
def e11():
    return load("e11_sensitivity")


@pytest.fixture(scope="module")
def e12():
    return load("e12_radius_profile")


def read(name):
    with open(ROOT / "results" / name, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# E11
# ---------------------------------------------------------------------------


def test_e11_csv_columns_and_size():
    rows = read("e11_sensitivity.csv")
    assert list(rows[0]) == ["polynomial", "method", "parameter", "value", "max_iter", "N", "half_width",
                             "converged (%)", "not converged (%)", "failed (%)", "mean iter", "mean evals"]
    # 4 polynomials x 5 methods x 10 settings (4 max_iter + 3 N + 3 window)
    assert len(rows) == 4 * 5 * 10
    assert {r["parameter"] for r in rows} == {"max_iter", "N", "window"}
    summary = read("e11_summary.csv")
    assert list(summary[0]) == ["polynomial", "parameter", "value", "ranking by mean evaluations",
                                "same as baseline ranking", "methods that change position"]
    assert len(summary) == 4 * 10
    baseline = [r for r in summary if r["parameter"] == "max_iter" and r["value"] == "baseline"]
    assert len(baseline) == 4 and all(r["same as baseline ranking"] == "true" for r in baseline)


def test_e11_baseline_rows_match_e2_and_e5():
    rows = [r for r in read("e11_sensitivity.csv") if r["parameter"] == "max_iter" and r["value"] == "baseline"]
    e2 = {(r["polynomial"], r["method"]): r for r in read("e2_cost.csv")}
    e2s = {(r["polynomial"], r["method"]): r for r in read("e2_summary.csv")}
    e5 = {(r["polynomial"], r["method"]): r for r in read("e5_polynomials.csv")}
    assert len(rows) == 20
    for r in rows:
        key = (r["polynomial"], r["method"])
        if key in e2:
            ref = (e2s[key]["converged (%)"], e2s[key]["not converged (%)"], e2s[key]["failed (%)"],
                   e2[key]["mean iter"], e2[key]["mean evals"])
        else:
            assert r["polynomial"] == "z^5 - 1"
            ref = (e5[key]["converged (%)"], e5[key]["not converged (%)"], e5[key]["failed (%)"],
                   e5[key]["mean iter"], e5[key]["mean evals"])
        mine = (r["converged (%)"], r["not converged (%)"], r["failed (%)"], r["mean iter"], r["mean evals"])
        assert [float(a) for a in mine] == [float(b) for b in ref], key


def test_e11_run_cell_matches_run_basins_directly(e11):
    from nfractals.methods import newton_step
    from nfractals.runner import CONVERGED, run_basins, single_point

    n, half = 60, 2.0
    r = e11.run_cell("z^3 - 1", "Newton", n, half)
    f, df, _, roots = e11.polynomial("z^3 - 1")
    z0 = e11.grid(n, half)
    _, iters, status = run_basins(z0, single_point(lambda z: newton_step(z, f, df)), roots, 1e-10, 50)
    assert r["converged"] == 100 * np.mean(status == CONVERGED)
    assert r["mean_iter"] == iters[status == CONVERGED].mean()
    assert r["max_iter"] == 50 and e11.run_cell("z^3 - 1", "Newton", n, half, 20)["max_iter"] == 20


def test_e11_ranking_orders_by_mean_evaluations_nan_last(e11):
    means = {"Newton": 15.0, "Halley": 13.0, "Damped Newton (h=0.7)": float("nan"), "Secant": 13.0,
             "Müller": 8.0}
    assert e11.ranking(means) == ["Müller", "Halley", "Secant", "Newton", "Damped Newton (h=0.7)"]


# ---------------------------------------------------------------------------
# E12
# ---------------------------------------------------------------------------


def test_e12_rings_partition_the_grid(e12):
    n = 60
    x = np.linspace(-2, 2, n)
    z0 = x[None, :] + 1j * x[:, None]
    ring = e12.rings_of(z0)
    assert ring.min() == 0 and ring.max() == 4
    counts = np.bincount(ring.ravel(), minlength=5)
    assert counts.sum() == n * n and (counts > 0).all()
    # edges: |z0| = 0.5 belongs to [0.5, 1), |z0| = 2 and the corners to the last ring
    assert e12.rings_of(np.array([0.5, 1.0, 1.5, 2.0, 2 * np.sqrt(2)])).tolist() == [1, 2, 3, 4, 4]
    assert e12.rings_of(np.array([0.0, 0.499, 1.999])).tolist() == [0, 0, 3]


def test_e12_ring_weighted_numbers_equal_the_global_ones_on_a_small_grid(e12):
    rows, glob, weighted = e12.ring_table("z^3 - 1", n=60)
    assert len(rows) == 5 * 5
    for name, (n_px, w_conv, w_evals) in weighted.items():
        assert n_px == 60 * 60
        assert w_conv == pytest.approx(glob[name][0], abs=1e-9)
        assert w_evals == pytest.approx(glob[name][1], abs=1e-9)
        assert sum(r[3] for r in rows if r[1] == name) == 60 * 60


def test_e12_csv_layout_and_baseline_numbers():
    rows = read("e12_radius_profile.csv")
    assert list(rows[0]) == ["polynomial", "method", "ring |z0|", "pixels", "converged (%)",
                             "not converged (%)", "failed (%)", "mean evals (converged only)"]
    assert len(rows) == 4 * 5 * 5
    e2s = {(r["polynomial"], r["method"]): r for r in read("e2_summary.csv")}
    for (poly, method), ref in e2s.items():
        rs = [r for r in rows if r["polynomial"] == poly and r["method"] == method]
        if not rs:
            continue  # z^3 - 1, z^3 - 2z + 2, z^6 - 1 and the random polynomial are in E12
        assert sum(int(r["pixels"]) for r in rs) == 1000 * 1000
        w = sum(int(r["pixels"]) * float(r["converged (%)"]) for r in rs) / 1e6
        assert w == pytest.approx(float(ref["converged (%)"]), abs=1e-4)
    assert (ROOT / "figures" / "e12_radius_profile.png").exists()
