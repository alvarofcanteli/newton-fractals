import numpy as np

from nfractals.methods import (  # noqa: E402
    halley_step,
    muller_init,
    muller_step,
    newton_step,
    secant_init,
    secant_step,
)
from nfractals.polynomials import POLYNOMIALS, make_polynomial  # noqa: E402
from nfractals.runner import CONVERGIDO, FALLO, NO_CONVERGE, run_basins, un_punto  # noqa: E402


def test_z0_2_converge_a_raiz_real_1():
    f, df, _, roots = make_polynomial(POLYNOMIALS["z^3 - 1"])
    z0 = np.array([[2 + 0j]])
    raiz, _, estado = run_basins(
        z0, un_punto(lambda z: newton_step(z, f, df)), f, roots, tol=1e-10, max_iter=100
    )
    assert estado[0, 0] == CONVERGIDO
    assert np.isclose(roots[raiz[0, 0]], 1)


def test_cuencas_de_z3_menos_1_son_un_tercio_en_el_disco():
    f, df, _, roots = make_polynomial(POLYNOMIALS["z^3 - 1"])
    x = np.linspace(-2, 2, 400)
    z0 = x[None, :] + 1j * x[:, None]
    en_disco = np.abs(z0) < 2
    z0 = np.where(en_disco, z0, np.nan)

    raiz, _, estado = run_basins(
        z0, un_punto(lambda z: newton_step(z, f, df)), f, roots, tol=1e-10, max_iter=100
    )
    convergidos = en_disco & (estado == CONVERGIDO)
    total = convergidos.sum()
    assert total > 0
    for k in range(3):
        frac = np.sum(convergidos & (raiz == k)) / total
        assert abs(frac - 1 / 3) <= 0.01, f"cuenca {k}: {frac:.4f}"


def test_metodo_con_estado_que_ignora_z_anterior_iguala_a_newton():
    f, df, _, roots = make_polynomial(POLYNOMIALS["z^3 - 1"])
    x = np.linspace(-2, 2, 101)
    z0 = x[None, :] + 1j * x[:, None]

    ref = run_basins(
        z0, un_punto(lambda z: newton_step(z, f, df)), f, roots, tol=1e-10, max_iter=50
    )
    con_estado = run_basins(
        z0,
        lambda s: (s[1], newton_step(s[1], f, df)),
        f,
        roots,
        tol=1e-10,
        max_iter=50,
        init=(z0, z0),
    )
    for esperado, obtenido in zip(ref, con_estado):
        np.testing.assert_array_equal(obtenido, esperado)


def test_secante_desde_z0_2_converge_a_raiz_1_en_z3_menos_1():
    f, _, _, roots = make_polynomial(POLYNOMIALS["z^3 - 1"])
    z0 = np.array([[2 + 0j]])
    raiz, _, estado = run_basins(
        z0,
        lambda s: secant_step(s, f),
        f,
        roots,
        tol=1e-10,
        max_iter=100,
        init=secant_init(z0),
    )
    assert estado[0, 0] == CONVERGIDO
    assert np.isclose(roots[raiz[0, 0]], 1)


def test_muller_desde_z0_2_converge_a_una_raiz_verdadera_en_z3_menos_1():
    f, _, _, roots = make_polynomial(POLYNOMIALS["z^3 - 1"])
    z0 = np.array([[2 + 0j]])
    raiz, _, estado = run_basins(
        z0,
        lambda s: muller_step(s, f),
        f,
        roots,
        tol=1e-10,
        max_iter=100,
        init=muller_init(z0),
    )
    assert estado[0, 0] == CONVERGIDO
    assert abs(f(roots[raiz[0, 0]])) < 1e-10


def test_ciclo_0_1_0_en_z3_menos_2z_mas_2():
    f, df, _, _ = make_polynomial(POLYNOMIALS["z^3 - 2z + 2"])
    z0 = np.array([0 + 0j])
    z1 = newton_step(z0, f, df)
    z2 = newton_step(z1, f, df)
    assert np.isclose(z1[0], 1)
    assert np.isclose(z2[0], 0)


def test_halley_desde_z_2_en_z3_menos_1():
    f, df, d2f, _ = make_polynomial(POLYNOMIALS["z^3 - 1"])
    # A mano: f = 8 - 1 = 7, f' = 3*4 = 12, f'' = 6*2 = 12
    # z1 = 2 - 2*7*12 / (2*12**2 - 7*12) = 2 - 168/204 = 20/17
    esperado = 20 / 17
    z1 = halley_step(np.array([2 + 0j]), f, df, d2f)
    assert np.isclose(z1[0], esperado)


def test_punto_0_en_z3_menos_1_es_fallo():
    f, df, _, roots = make_polynomial(POLYNOMIALS["z^3 - 1"])
    z0 = np.array([[0 + 0j]])
    _, _, estado = run_basins(
        z0, un_punto(lambda z: newton_step(z, f, df)), f, roots, tol=1e-10, max_iter=50
    )
    assert estado[0, 0] == FALLO


def test_punto_0_en_z3_menos_2z_mas_2_no_converge():
    f, df, _, roots = make_polynomial(POLYNOMIALS["z^3 - 2z + 2"])
    z0 = np.array([[0 + 0j]])
    _, _, estado = run_basins(
        z0, un_punto(lambda z: newton_step(z, f, df)), f, roots, tol=1e-10, max_iter=50
    )
    assert estado[0, 0] == NO_CONVERGE
