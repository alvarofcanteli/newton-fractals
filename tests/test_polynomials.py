import numpy as np
import pytest

from nfractals.polynomials import unique_roots, unity_poly

# ---------------------------------------------------------------------------
# unity_poly(n): coefficients of z^n - 1, from highest to lowest degree
# ---------------------------------------------------------------------------


def test_unity_poly_3():
    assert unity_poly(3) == [1, 0, 0, -1]


def test_unity_poly_1():
    assert unity_poly(1) == [1, -1]


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_unity_poly_has_n_plus_1_coefficients(n):
    assert len(unity_poly(n)) == n + 1


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_unity_poly_roots_are_the_nth_roots_of_unity(n):
    roots = np.roots(unity_poly(n))
    assert len(roots) == n
    assert np.allclose(np.abs(roots), 1)
    assert np.allclose(roots**n, 1)


# ---------------------------------------------------------------------------
# unique_roots(roots, tol): merge roots closer than tol (by averaging)
# ---------------------------------------------------------------------------


def test_unique_roots_keeps_distinct_roots():
    roots = np.roots([1, 0, 0, -1])  # z^3 - 1
    result = unique_roots(roots)
    assert len(result) == 3
    np.testing.assert_allclose(result, roots)


def test_unique_roots_merges_double_root():
    roots = np.roots([1, -1, -1, 1])  # (z - 1)^2 (z + 1)
    result = unique_roots(roots)
    assert len(result) == 2
    np.testing.assert_allclose(np.sort(result.real), [-1, 1], atol=1e-6)
    np.testing.assert_allclose(result.imag, 0, atol=1e-6)


def test_unique_roots_averages_merged_roots():
    result = unique_roots([1, 1 + 2e-7])
    np.testing.assert_allclose(result, [1 + 1e-7], rtol=0, atol=1e-15)


def test_unique_roots_grouping_is_transitive():
    # 0 and 1.2e-6 are farther apart than tol, but both are close to 0.6e-6
    result = unique_roots([0, 0.6e-6, 1.2e-6], tol=1e-6)
    assert len(result) == 1
    np.testing.assert_allclose(result, [0.6e-6], rtol=0, atol=1e-15)


def test_unique_roots_keeps_order_of_first_appearance():
    result = unique_roots([5, 1, 5 + 1e-7])
    np.testing.assert_allclose(result, [5 + 0.5e-7, 1], rtol=0, atol=1e-12)


def test_unique_roots_respects_tol_argument():
    assert len(unique_roots([0, 0.5])) == 2
    assert len(unique_roots([0, 0.5], tol=1)) == 1


def test_random_poly_is_reproducible_and_in_range():
    from nfractals.polynomials import random_poly

    c1, r1 = random_poly()
    c2, r2 = random_poly()
    assert np.array_equal(c1, c2) and np.array_equal(r1, r2)
    assert len(c1) == 6 and c1[0] == 1
    assert np.all((np.abs(r1) >= 0.3) & (np.abs(r1) < 1.5))
    assert np.allclose(np.sort_complex(np.roots(c1)), np.sort_complex(r1))
