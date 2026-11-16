"""Tests of the extension of E2, E4 and E7 to more polynomials (CSV layout, z^3 - 1 rows unchanged,
E4 starting points)."""
import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
POLYS = ["z^3 - 1", "z^3 - 2z + 2", "z^6 - 1", "random degree 5"]


def read(name):
    with open(ROOT / "results" / name, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.mark.parametrize("name", ["e2_cost.csv", "e4_order.csv", "e7_precision.csv"])
def test_polynomial_column_is_first(name):
    with open(ROOT / "results" / name, newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    assert header[0] == "polynomial"


@pytest.mark.parametrize("name", ["e2_cost.csv", "e7_precision.csv"])
def test_every_polynomial_has_five_methods(name):
    rows = read(name)
    for poly in POLYS:
        assert len([r for r in rows if r["polynomial"] == poly]) == 5


@pytest.mark.parametrize("name", ["e2_summary.csv", "e7_summary.csv"])
def test_summary_has_one_row_per_polynomial_and_method(name):
    rows = read(name)
    keys = {(r["polynomial"], r["method"]) for r in rows}
    assert len(rows) == len(keys) == 20


def test_e4_summary_has_one_row_per_polynomial_and_method():
    rows = read("e4_summary.csv")
    keys = {(r["polynomial"], r["method"]) for r in rows}
    assert len(rows) == len(keys) == 21  # 5 + 5 + 5 + 5 simple-root cases and the double root
    assert {r["polynomial"] for r in rows} == set(POLYS) | {"(z-1)^2 (z+1)"}


def test_e2_z3_minus_1_rows_unchanged():
    expected = {  # method: (mean iter, mean evals), values of the README / earlier CSV
        "Newton": ("7.586", "15.172"),
        "Halley": ("4.39", "13.169"),
        "Damped Newton (h=0.7)": ("21.841", "43.683"),
        "Secant": ("10.253", "12.253"),
        "Müller": ("5.613", "8.613"),
    }
    rows = {r["method"]: r for r in read("e2_cost.csv") if r["polynomial"] == "z^3 - 1"}
    for method, (it, ev) in expected.items():
        assert (rows[method]["mean iter"], rows[method]["mean evals"]) == (it, ev)


def test_e7_z3_minus_1_rows_unchanged():
    expected = {  # method: root differs (% of all)
        "Newton": "0.0026", "Halley": "0.0", "Damped Newton (h=0.7)": "0.001",
        "Secant": "0.155", "Müller": "0.0548",
    }
    rows = {r["method"]: r for r in read("e7_precision.csv") if r["polynomial"] == "z^3 - 1"}
    for method, value in expected.items():
        assert rows[method]["root differs (% of all)"] == value
    assert rows["Müller"]["delta"] == "0.01" and rows["Secant"]["delta"] == "0.001"


def test_e4_original_cases_unchanged():
    rows = read("e4_order.csv")
    newton = [r for r in rows if r["polynomial"] == "z^3 - 1" and r["method"] == "Newton (z^3-1)"]
    assert newton[1]["e_k"] == "0.00234315948601663" and newton[1]["p_k"] == "1.979849666"
    double = [r for r in rows if r["polynomial"] == "(z-1)^2 (z+1)"]
    assert double and all(r["method"] == "Newton at double root ((z-1)^2 (z+1))" for r in double)


def test_e4_starting_points_converge_to_a_simple_root_with_the_five_methods():
    spec = importlib.util.spec_from_file_location("e4_order", ROOT / "experiments" / "e4_order.py")
    e4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e4)  # importing does not run the experiment (main guard)
    # run() asserts for every new case: z0 nearest to its root, simple root, the five methods
    # reach TOL and no iterate leaves the half-distance to the nearest other root.
    _, summary, roots_of, starts = e4.run()
    for name, z0 in e4.NEW_CASES:
        rows = [s for s in summary if s[0] == name]
        assert len(rows) == 5
        coefs = np.asarray(e4.POLYNOMIALS[name], dtype=complex)
        r = complex(roots_of[name])
        assert abs(np.polyval(np.polyder(coefs), r)) > 0.1  # simple root
        assert np.min(np.abs(np.roots(coefs) - r)) < 1e-12  # it is a root of the polynomial
        assert np.isclose(abs(np.polyval(coefs, r)), 0, atol=1e-12)
        assert complex(starts[name]) == complex(z0)
        for _, _, es, _, _, _, _, _ in rows:
            assert es[-1] < e4.TOL


def test_e7_summary_counts_columns_exist_and_are_consistent():
    rows = read("e7_summary.csv")
    new = ["n_both_converged_changed", "n_both_converged_changed_boundary", "n_single_precision",
           "n_single_precision_boundary_b", "n_single_precision_near_nonconv_3px"]
    assert all(c in rows[0] for c in new)
    assert "z^3 - 1" in {r["polynomial"] for r in rows}
    for r in rows:
        changed, changed_bd, single, single_b, single_near = (int(r[c]) for c in new)
        assert 0 <= changed_bd <= changed
        assert 0 <= single_b <= single and 0 <= single_near <= single
        # the counts agree with the percentages already in the summary (N = 1000)
        assert changed == int(r["root differs (pixels)"])
        assert single == int(r["converged in only one precision (pixels)"])
