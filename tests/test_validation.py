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


def test_boundary_fraction_of_two_half_planes_is_about_two_over_n():
    from nfractals.metrics import boundary_fraction
    from nfractals.runner import CONVERGED

    n = 100
    root = np.zeros((n, n), dtype=int)
    root[:, n // 2:] = 1  # vertical straight boundary: two columns of boundary pixels
    status = np.full((n, n), CONVERGED)
    assert np.isclose(boundary_fraction(root, status), 2 / n)
    # Non-converged pixels are not counted.
    status[: n // 2, :] = 0
    assert np.isclose(boundary_fraction(root, status), 1 / n)

# ---------------------------------------------------------------------------
# E10: non-polynomial functions and the callable interface of run_basins
# ---------------------------------------------------------------------------


def test_sin_nearest_root_and_id():
    from nfractals.functions import sin_nearest

    z = np.array([0.1, 3.0 + 0.5j, -1.0, -5.0, 1.5 + 10j, 2 * np.pi + 0.2j])
    ids, dist = sin_nearest(z)
    assert ids.tolist() == [0, 1, 0, -2, 0, 2]
    assert np.allclose(dist, np.abs(z - ids * np.pi))


def test_lambert_roots_are_roots_of_exp_minus_z():
    from nfractals.functions import lambert_roots

    ks, roots = lambert_roots()
    assert len(roots) == 41
    assert np.all(np.abs(np.exp(roots) - roots) < 1e-12)
    # in [-3, 3]^2 the roots are 0.318 +- 1.337i
    inside = roots[(np.abs(roots.real) < 3) & (np.abs(roots.imag) < 3)]
    assert len(inside) == 2 and np.allclose(sorted(inside.imag), [-1.3372, 1.3372], atol=1e-4)
    assert np.allclose(inside.real, 0.3181, atol=1e-4)


def test_newton_on_sin_from_0_1_converges_to_zero():
    from nfractals.functions import sin_nearest

    z0 = np.array([[0.1 + 0j]])
    root, iters, status = run_basins(
        z0, single_point(lambda z: newton_step(z, np.sin, np.cos)), sin_nearest, 1e-10, 50)
    assert status[0, 0] == CONVERGED and root[0, 0] == 0 and iters[0, 0] <= 6


def test_run_basins_with_callable_matches_array_of_roots():
    f, df, _, roots_np = make_polynomial(POLYNOMIALS["z^3 - 1"])
    roots = unique_roots(roots_np)

    def nearest(z):
        d = np.abs(z[:, None] - roots[None, :])
        j = np.argmin(d, axis=1)
        return j, d[np.arange(j.size), j]

    x = np.linspace(-2, 2, 40)
    z0 = x[None, :] + 1j * x[:, None]
    step = single_point(lambda z: newton_step(z, f, df))
    a = run_basins(z0, step, roots, 1e-10, 50)
    b = run_basins(z0, step, nearest, 1e-10, 50)
    assert all(np.array_equal(u, v) for u, v in zip(a, b))
