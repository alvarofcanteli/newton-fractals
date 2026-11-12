# Fractales de Newton y métodos de búsqueda de raíces

Este proyecto calcula y compara las **cuencas de atracción** de varios métodos de búsqueda de raíces
(Newton, Halley, Newton amortiguado, secante y Müller) sobre polinomios en el plano complejo.
Para cada punto de una malla se itera el método, se clasifica el resultado (convergido a una raíz,
no converge o fallo numérico) y se mide el coste y la forma de las cuencas.

**Pregunta guía:** ¿cómo cambian la forma de las cuencas, el coste y la robustez de un método de
búsqueda de raíces al cambiar el método, el polinomio, el amortiguamiento o la precisión aritmética?

## Métodos

- **Newton:** `z - f(z)/f'(z)`.
- **Halley:** `z - 2 f f' / (2 f'^2 - f f'')`.
- **Newton amortiguado:** `z - h f(z)/f'(z)` con `h = 0,7` (salvo en E6, que varía `h`).
- **Secante:** usa `(z_{k-1}, z_k)`, con la convención `z_{-1} = z_0 + δ`.
- **Müller:** interpola una parábola por `(z_{k-2}, z_{k-1}, z_k)` y elige el signo de la raíz cuadrada
  que da el denominador de mayor módulo; estado inicial `(z_0, z_0 + δ, z_0 + 2δ)`.

## Experimentos

| Experimento | Qué hace | Script |
|---|---|---|
| E0 | Mapa de cuencas de Newton en z³−1 y comparación de los cinco métodos | `experiments/e0_newton_z3.py`, `experiments/e0_compare.py` |
| E1 | % de cuencas por raíz según el tamaño de malla N y la ventana (cuadrado o disco) | `experiments/e1_basins.py` |
| E2 | Coste: iteraciones y evaluaciones de f/f'/f'' hasta converger, con histogramas | `experiments/e2_cost.py` |
| E3 | Fallos y no convergencia en z³−2z+2 (ciclo 0 → 1 → 0 de Newton) | `experiments/e3_failures.py` |
| E3b | Diagnóstico de la secante en z³−1: no convergidos según `max_iter` y su estado final (genera `figures/diag_secante.png`) | `experiments/e3b_secante_diag.py` |
| E4 | Orden de convergencia con mpmath a 300 dígitos, incluida la raíz doble | `experiments/e4_order.py` |
| E5 | Comparación en z³−1, z⁴−1, z⁵−1, z⁶−1, (z−1)²(z+1) y z³−2z+2 | `experiments/e5_polynomials.py` |
| E5b | Comprobaciones sobre E5: Newton en z⁵−1 con `max_iter` = 40 y 50, y dónde falla la secante (dentro y fuera de \|z₀\| < 1) | `experiments/e5b_checks.py` |
| E6 | Newton amortiguado en función de `h` y complejidad de la frontera | `experiments/e6_damped.py` |
| E7 | Efecto de la precisión: complex64 frente a complex128 | `experiments/e7_precision.py` |
| E7b | Müller en E7: efecto de δ y distancia de los píxeles que difieren a las fronteras (genera `figures/e7_diff_muller.png`) | `experiments/e7b_muller_delta.py` |

## Instalación y ejecución

Requiere Python 3.9 o superior. Desde la raíz del repositorio:

```bash
python -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate
pip install -e .
```

Tests:

```bash
python -m pytest
```

Experimentos (cada uno escribe en `results/` y `figures/`; E3 y E5 tardan varios minutos):

```bash
python experiments/e0_newton_z3.py
python experiments/e0_compare.py
python experiments/e1_basins.py
python experiments/e2_cost.py
python experiments/e3_failures.py
python experiments/e3b_secante_diag.py
python experiments/e4_order.py
python experiments/e5_polynomials.py
python experiments/e5b_checks.py
python experiments/e6_damped.py
python experiments/e7_precision.py
python experiments/e7b_muller_delta.py
```

## Estructura de carpetas

```
src/nfractals/
    methods.py        pasos de Newton, Halley, secante y Müller (NumPy, vectorizados)
    runner.py         run_basins: itera la malla con estado por método y clasifica los puntos
    polynomials.py    polinomios, derivadas, raíces de referencia (unique_roots, unity_poly)
    plotting.py       mapas de cuencas (matiz = raíz, brillo = iteraciones)
    convergence.py    versiones escalares con mpmath para medir órdenes de convergencia
experiments/          un script por experimento (E0 a E7) y los de diagnóstico (E3b, E5b, E7b)
tests/                tests de validación (pytest)
results/              tablas CSV de los experimentos
figures/              figuras generadas
```

## Resultados en z³−1

Ventana [−2, 2]², N = 1000, tol = 1e-10 (complex128). Los datos salen de `results/e2_cost.csv`
(iteraciones y evaluaciones), `results/e5_polynomials.csv` (% de convergencia) y
`results/e4_order.csv` (orden observado: media de los tres últimos `p_k` de la serie con
300 dígitos).

| Método | max_iter | Iteraciones medias | Evaluaciones medias | % convergido | Orden observado |
|---|---|---|---|---|---|
| Newton | 50 | 7,652 | 15,304 | 100,00 | 2,000 |
| Halley | 50 | 4,429 | 13,286 | 100,00 | 2,996 (tiende a 3) |
| Newton amortiguado (h = 0,7) | 200 | 22,734 | 45,469 | 100,00 | 1 (razón 0,3) |
| Secante | 100 | 10,348 | 12,348 | 94,74 | 1,617 |
| Müller | 100 | 5,688 | 8,688 | 100,00 | 1,845 |

Las iteraciones y evaluaciones medias son las de los puntos convergidos.

![Cuencas de z³−1 con los cinco métodos](figures/e0_compare.png)

*Cuencas de z³−1 en [−2, 2]², de izquierda a derecha: Newton, Halley, Newton amortiguado, secante y
Müller. Cada raíz tiene un color (rojo, verde, azul) y el brillo baja al aumentar las iteraciones;
el negro marca los puntos que no convergen. Las fronteras de Newton y del amortiguado son
fractales, con islas de otro color. Las de Halley son más lisas y con menos islas. La secante
muestra bandas fractales anchas a lo largo de las fronteras, con zonas negras (no convergen),
sobre todo cerca del centro. Las fronteras de Müller son casi rectas, con tres sectores, y solo
aparecen detalles pequeños en los bordes de la ventana.*

![Newton amortiguado: iteraciones medias y fracción de frontera frente a h](figures/e6_curves.png)

*Newton amortiguado en z³−1 y z³−2z+2. A la izquierda, las iteraciones medias de los convergidos
bajan de unas 70 con h = 0,3 a unas 7,5 con h = 1, casi igual en los dos polinomios. A la derecha,
la fracción de píxeles de frontera sube con h: en z³−1 de aproximadamente 1,5 % a 3,5 %, y en
z³−2z+2 de aproximadamente 1,2 % a 2,8 %, siempre por debajo de z³−1.*

![Fallos y no convergencia en z³−2z+2](figures/e3_failures.png)

*Cuencas de z³−2z+2 en [−2, 2]² con los mismos cinco métodos y orden que la figura anterior.
La raíz real (−1,769) ocupa en rojo la zona izquierda y las dos complejas el resto, en verde y
azul. Newton tiene discos negros sobre el eje real (los puntos que no convergen por el ciclo
0 → 1 → 0). Halley y el amortiguado no tienen puntos negros. La secante solo muestra una pequeña
mancha negra junto al eje real. Müller tiene fronteras más lisas y redondeadas.*

## Convenciones y parámetros

- **Ventana y malla:** [−2, 2]² con N × N puntos (N = 1000, salvo en E1, que usa 250, 500, 1000 y 2000).
- **Criterio de parada:** un punto converge cuando `|f(z)| < tol` en la iteración actual. Si el
  nuevo valor no es finito (inf o NaN), el punto se marca como **fallo**; si se agota `max_iter`,
  como **no converge**. La raíz asignada es la más cercana (`argmin |z − raíz|`) entre las raíces
  distintas de `np.roots`, agrupando con `unique_roots` las raíces casi idénticas (raíces dobles).
- **Tolerancias:** `1e-10` en complex128; `1e-5` en complex64 (E7, unas 84 veces el épsilon de
  máquina de 1,2e-7); `1e-8` en el polinomio de raíz doble (E5). E4 trabaja con mpmath a 300 dígitos
  y se detiene al bajar el error de `1e-250`.
- **`max_iter`:** 50 en Newton y Halley, 200 en Newton amortiguado (300 en E6), 100 en secante
  y Müller.
- **δ de secante y Müller:** `δ = 1e-3` en todos los experimentos.
- **Coste por paso y evaluaciones iniciales:**
  Newton 2 evaluaciones por paso (f, f'), Halley 3 (f, f', f''), Newton amortiguado 2,
  secante 1 más 2 iniciales, Müller 1 más 3 iniciales. Evaluaciones por punto =
  iteraciones × coste por paso + evaluaciones iniciales.

## Equipo y asignatura

- Pablo Arístegui
- Alfonso Botín-Sanz de Sautuola
- Álvaro Fernández-Canteli

Asignatura: Numerical Methods, IE University, curso 2026-27.
