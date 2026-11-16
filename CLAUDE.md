# CLAUDE.md

## Language

All code, comments, docstrings, commit messages, README, figures, report and presentation text must be
in English, whatever language the user writes to you in.

## Project

Numerical Methods final project (IE University): "Newton fractals: basins of attraction and reliability
of rootfinding methods". Methods: Newton, Halley, damped Newton, secant and Müller.

- `src/nfractals/`: the package (`methods.py`, `runner.py`, `polynomials.py`, `plotting.py`, `metrics.py`, `functions.py`, `convergence.py`).
- `experiments/`: one script per experiment, E0–E12 (including `e11_sensitivity` and `e12_radius_profile`), plus the diagnostic scripts E3b, E5b, E5c and E7b.
- `results/`: CSV output of the experiments. `figures/`: generated figures.
- `tests/`: pytest validation tests. Install with `pip install -e .`.

## Conventions

- **Brightness scale:** basin plots use a shared `iter_ref` (99th percentile of the iterations of the
  converged points of all methods in the same figure), so brightness is comparable across panels.
- **Colours:** one hue per root; non-converged (and failed) points are drawn in black.
- **Statuses:** `ACTIVE` / `CONVERGED` / `FAILED` / `NOT_CONVERGED` (see `runner.py`). A point converges
  when `|z - r| < tol` for its nearest distinct root `r`; a non-finite value is `FAILED`; running out of `max_iter` is `NOT_CONVERGED`.
  `run_basins` also accepts a nearest-root callable instead of an array of roots (E10, non-polynomial functions, an extension beyond the course guide).
- **Cost:** counted as evaluations of f (and its derivatives), including the initial evaluations of each
  method: Newton 2 per step, Halley 3, damped Newton 2, secant 1 per step + 2 initial, Müller 1 per step + 3 initial.
- **Means** (iterations, evaluations) are computed only over converged points.
- **Initial points of secant and Müller:** secant starts from `(z0 + δ, z0)`, Müller from
  `(z0, z0 + δ, z0 + 2δ)`, with `δ = 1e-3` unless an experiment says otherwise (E7 uses 1e-2 for Müller; E7b sweeps 1e-1, 1e-2 and 1e-3).
  Always state δ when reporting results for these methods.
- **Numbers:** use a decimal point, not a comma (7.652, not 7,652).

## Rules

- Run `python -m pytest` before every commit.
- Never change numerical results without telling the user.
- Do not push unless the user explicitly asks.
- No local absolute paths in the repository (nothing like `C:\Users\...`).
- Present results to the user as plain-text lists, not box-drawing tables.

## Pitfalls to avoid

- Never claim the basins of zⁿ−1 must be equal in a square window; that holds only in a disk (the square breaks the symmetry).
- Never use the same tolerance in float32 and float64 (E7 uses 1e-5 and 1e-10), nor a tolerance that is unreachable for multiple roots (E5 uses 1e-6 for the double root; 1e-8 is at the numerical floor under the `|z - r|` criterion).
- Never estimate the order of convergence in double precision when the error is already at the numerical floor; use mpmath (E4).
- Never compare methods by iterations only; also count evaluations of f.
- Always document the initial-points convention (δ) of secant and Müller.
- Never draw conclusions from a single grid resolution (E1 varies N).
- Never state conclusions the metrics do not support.

## Report and presentation

- Scientific-article structure: abstract, introduction and question, foundations, methodology (parameters and
  conventions), results per experiment, discussion (explain why, not only what), limitations, conclusions, references.
- Every figure caption states the window, the parameters and how to read the colours.
- Verify every reference before citing it.
- Write in English.

## Definition of done for an experiment

A script in `experiments/`, its CSV in `results/`, its figures in `figures/`, a row in the README
experiments table, and `python -m pytest` passing.

## Team workflow

- Run `git pull` before starting.
- Small commits with clear messages; one branch per task when working in parallel.
- Never force-push; never rewrite pushed history.

## Working style

- Before changing core modules (`runner.py`, `methods.py`, `polynomials.py`), explain the change first.
- Ask before deleting or renaming files.
- Do not add dependencies without asking.

## Team decisions

- **Stopping criterion:** `|z - r| < tol`, where `r` is the nearest distinct root (from `unique_roots`); that
  root is the one assigned to the point. Tolerances: 1e-10 in complex128, 1e-5 in complex64 and 1e-6 for the
  double-root polynomial (see the pitfalls).
- **δ:** `1e-3` everywhere, except E7, where Müller uses `δ = 1e-2` in both complex64 and complex128 (secant
  stays at `1e-3`). E7b keeps its sweep over `1e-1`, `1e-2` and `1e-3` as the diagnostic.
