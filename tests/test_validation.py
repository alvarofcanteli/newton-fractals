import numpy as np

from nfractals.methods import (  # noqa: E402
    halley_step,
    muller_init,
    muller_step,
    newton_step,
    secant_init,
    secant_step,
)
from nfractals.polynomials import POLYNOMIALS, make_polynomial, unique_roots  # noqa: E402
from nfractals.runner import (  # noqa: E402
    CONVERGED,
    FAILED,
    NOT_CONVERGED,
    run_basins,
    single_point,
)


def test_z0_2_converges_to_real_root_1():
    f, df, _, roots = make_polynomial(POLYNOMIALS["z^3 - 1"])
    z0 = np.array([[2 + 0j]])
    root, _, status = run_basins(
        z0, single_point(lambda z: newton_step(z, f, df)), roots, tol=1e-10, max_iter=100
    )
    assert status[0, 0] == CONVERGED
    assert np.isclose(roots[root[0, 0]], 1)


def test_basins_of_z3_minus_1_are_one_third_in_the_disk():
    f, df, _, roots = make_polynomial(POLYNOMIALS["z^3 - 1"])
    x = np.linspace(-2, 2, 400)
    z0 = x[None, :] + 1j * x[:, None]
    in_disk = np.abs(z0) < 2
    z0 = np.where(in_disk, z0, np.nan)

    root, _, status = run_basins(
        z0, single_point(lambda z: newton_step(z, f, df)), roots, tol=1e-10, max_iter=100
    )
    converged = in_disk & (status == CONVERGED)
    total = converged.sum()
    assert total > 0
    for k in range(3):
        frac = np.sum(converged & (root == k)) / total
        assert abs(frac - 1 / 3) <= 0.01, f"basin {k}: {frac:.4f}"


def test_stateful_method_that_ignores_previous_z_matches_newton():
    f, df, _, roots = make_polynomial(POLYNOMIALS["z^3 - 1"])
    x = np.linspace(-2, 2, 101)
    z0 = x[None, :] + 1j * x[:, None]

    ref = run_basins(
        z0, single_point(lambda z: newton_step(z, f, df)), roots, tol=1e-10, max_iter=50
    )
    with_state = run_basins(
        z0,
        lambda s: (s[1], newton_step(s[1], f, df)),
        roots,
        tol=1e-10,
        max_iter=50,
        init=(z0, z0),
    )
    for expected, obtained in zip(ref, with_state):
        np.testing.assert_array_equal(obtained, expected)


def test_secant_from_z0_2_converges_to_root_1_in_z3_minus_1():
    f, _, _, roots = make_polynomial(POLYNOMIALS["z^3 - 1"])
    z0 = np.array([[2 + 0j]])
    root, _, status = run_basins(
        z0,
        lambda s: secant_step(s, f),
        roots,
        tol=1e-10,
        max_iter=100,
        init=secant_init(z0),
    )
    assert status[0, 0] == CONVERGED
    assert np.isclose(roots[root[0, 0]], 1)


def test_muller_from_z0_2_converges_to_a_true_root_in_z3_minus_1():
    f, _, _, roots = make_polynomial(POLYNOMIALS["z^3 - 1"])
    z0 = np.array([[2 + 0j]])
    root, _, status = run_basins(
        z0,
        lambda s: muller_step(s, f),
        roots,
        tol=1e-10,
        max_iter=100,
        init=muller_init(z0),
    )
    assert status[0, 0] == CONVERGED
    assert abs(f(roots[root[0, 0]])) < 1e-10


def test_cycle_0_1_0_in_z3_minus_2z_plus_2():
    f, df, _, _ = make_polynomial(POLYNOMIALS["z^3 - 2z + 2"])
    z0 = np.array([0 + 0j])
    z1 = newton_step(z0, f, df)
    z2 = newton_step(z1, f, df)
    assert np.isclose(z1[0], 1)
    assert np.isclose(z2[0], 0)


def test_halley_from_z_2_in_z3_minus_1():
    f, df, d2f, _ = make_polynomial(POLYNOMIALS["z^3 - 1"])
    # By hand: f = 8 - 1 = 7, f' = 3*4 = 12, f'' = 6*2 = 12
    # z1 = 2 - 2*7*12 / (2*12**2 - 7*12) = 2 - 168/204 = 20/17
    expected = 20 / 17
    z1 = halley_step(np.array([2 + 0j]), f, df, d2f)
    assert np.isclose(z1[0], expected)


def test_point_0_in_z3_minus_1_is_failure():
    f, df, _, roots = make_polynomial(POLYNOMIALS["z^3 - 1"])
    z0 = np.array([[0 + 0j]])
    _, _, status = run_basins(
        z0, single_point(lambda z: newton_step(z, f, df)), roots, tol=1e-10, max_iter=50
    )
    assert status[0, 0] == FAILED


def test_point_0_in_z3_minus_2z_plus_2_does_not_converge():
    f, df, _, roots = make_polynomial(POLYNOMIALS["z^3 - 2z + 2"])
    z0 = np.array([[0 + 0j]])
    _, _, status = run_basins(
        z0, single_point(lambda z: newton_step(z, f, df)), roots, tol=1e-10, max_iter=50
    )
    assert status[0, 0] == NOT_CONVERGED


def test_convergence_means_distance_to_the_nearest_root_below_tol():
    # The step always returns the same point, at distance 5e-11 / 5e-9 from the root 1.
    roots = np.array([1 + 0j, -1 + 0j])
    z0 = np.array([[0.3 + 0j]])
    close = run_basins(z0, single_point(lambda z: np.full_like(z, 1 + 5e-11)), roots, tol=1e-10, max_iter=3)
    far = run_basins(z0, single_point(lambda z: np.full_like(z, 1 + 5e-9)), roots, tol=1e-10, max_iter=3)
    assert close[2][0, 0] == CONVERGED and close[1][0, 0] == 1 and close[0][0, 0] == 0
    assert far[2][0, 0] == NOT_CONVERGED and far[1][0, 0] == 0 and far[0][0, 0] == -1


def test_small_residual_is_not_enough_at_a_double_root():
    # (z-1)^2 (z+1) at z = 1 + 1e-6: |f| ~ 2e-12 < 1e-10 (old criterion) but |z - 1| = 1e-6 > 1e-10.
    f, _, _, r = make_polynomial(POLYNOMIALS["(z-1)^2 (z+1)"])
    roots = unique_roots(r)
    z = np.array([[1 + 1e-6 + 0j]])
    assert abs(f(z)[0, 0]) < 1e-10
    _, _, status = run_basins(z, single_point(lambda w: w), roots, tol=1e-10, max_iter=2)
    assert status[0, 0] == NOT_CONVERGED
