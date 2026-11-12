# CLAUDE.md

## Language

All code, comments, docstrings, commit messages, README, figures, report and presentation text must be
in English, whatever language the user writes to you in.

## Project

Numerical Methods final project (IE University): "Newton fractals: basins of attraction and reliability
of rootfinding methods". Methods: Newton, Halley, damped Newton, secant and Müller.

- `src/nfractals/`: the package (`methods.py`, `runner.py`, `polynomials.py`, `plotting.py`, `convergence.py`).
- `experiments/`: one script per experiment, E0–E7, plus the diagnostic scripts E3b, E5b and E7b.
- `results/`: CSV output of the experiments. `figures/`: generated figures.
- `tests/`: pytest validation tests. Install with `pip install -e .`.

## Conventions

- **Brightness scale:** basin plots use a shared `iter_ref` (99th percentile of the iterations of the
  converged points of all methods in the same figure), so brightness is comparable across panels.
- **Colours:** one hue per root; non-converged (and failed) points are drawn in black.
- **Statuses:** `ACTIVE` / `CONVERGED` / `FAILED` / `NOT_CONVERGED` (see `runner.py`). A point converges
  when `|f(z)| < tol`; a non-finite value is `FAILED`; running out of `max_iter` is `NOT_CONVERGED`.
- **Cost:** counted as evaluations of f (and its derivatives), including the initial evaluations of each
  method: Newton 2 per step, Halley 3, damped Newton 2, secant 1 per step + 2 initial, Müller 1 per step + 3 initial.
- **Means** (iterations, evaluations) are computed only over converged points.
- **Initial points of secant and Müller:** secant starts from `(z0 + δ, z0)`, Müller from
  `(z0, z0 + δ, z0 + 2δ)`, with `δ = 1e-3` unless an experiment says otherwise (E7b tries 1e-2 and 1e-1).
  Always state δ when reporting results for these methods.
- **Numbers:** use a decimal point, not a comma (7.652, not 7,652).

## Rules

- Run `python -m pytest` before every commit.
- Never change numerical results without telling the user.
- Do not push unless the user explicitly asks.
- No local absolute paths in the repository (nothing like `C:\Users\...`).
- Present results to the user as plain-text lists, not box-drawing tables.
