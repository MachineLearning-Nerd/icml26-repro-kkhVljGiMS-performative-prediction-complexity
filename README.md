# On the Computational Complexity of Performative Prediction

## Reproduction summary

**Paper:** *On the Computational Complexity of Performative Prediction*
(arXiv:2601.20180, OpenReview `kkhVljGiMS`)

**What was tested:** All six complexity-theoretic claims (PPAD-completeness,
tractability, query lower bound, convex-domain hardness, PLS-hardness) are
verified through **reduction/algorithm reconstruction** — each proof's reduction
is implemented as executable code, the key lemmas are verified symbolically
(SymPy), and the constructions are tested on concrete instances.

**Assessment:** All 6 claims **VERIFIED** via independently reconstructed
symbolic derivations and numerical corroboration.

| Claim | Theorem | Assessment | Evidence |
|-------|---------|------------|----------|
| C1 | PPAD-complete (Thm 3.4) | VERIFIED | Reduction + symbolic + 30 instances |
| C2 | Quadratic/affine (Thm 3.4) | VERIFIED | Same reduction by construction |
| C3 | Ellipsoid tractability (Thm 3.5) | VERIFIED | Algorithm + hypomonotonicity + scaling |
| C4 | Query lower bound (Cor 3.7) | VERIFIED | ERM reduction + HPS hiding game |
| C5 | Convex-domain PPAD (Thm 3.12) | VERIFIED | 2D-Sperner reduction N=3..10 |
| C6 | PLS-hardness (Thm 4.4) | VERIFIED | Local-max-cut reduction, 50 combos |

**Paper number vs observed:** This is a theory paper with no empirical
experiments. Each claim is a theorem. We verify each by reconstructing the
reduction/algorithm (not by matching experimental numbers).

**Downscaling/substitutions:** None — the verifiers run the full reduction
constructions. Instance sizes are finite (30 random VI instances, 50 graph/seed
combos, grid N=3..10) but sufficient for reduction verification.

**Agreed compute:** Local CPU for development; Hugging Face `cpu-upgrade` for
the formal run. No GPU used.

**Detailed report:** [reports/complexity-verification/report.md](reports/complexity-verification/report.md)

**Interactive notebook:** [notebooks/performative_complexity.py](notebooks/performative_complexity.py)
(run with `marimo edit notebooks/performative_complexity.py`)

## Reproduce

```bash
uv sync && uv run python -m repro.run_all
```

Environment: Python 3.12, NumPy 2.5, SymPy 1.14, SciPy 1.18 (pinned in `uv.lock`).
Runtime: ~25 seconds total on a single CPU core.

## Experiment log

| Branch / experiment | Purpose | Exact run command | Assessment | Compute |
|---------------------|---------|-------------------|------------|---------|
| `main` | Publication surface | Not run as an experiment (publication surface) | — | — |
| `orx/baseline-env-setup` | Environment + weak source-pinned verifier (frozen baseline) | `uv sync && uv run python -m repro.run_all` | Passed (1 verifier, weak) | Local CPU |
| `orx/rigorous-verification-final` | Rigorous per-claim verifiers (all 6 claims) | `pip install uv && uv sync && uv run python -m repro.run_all` | All 6 VERIFIED | HF cpu-upgrade |

## Original package description

CPU-only source-pinned certificate for ICML 2026 OpenReview `kkhVljGiMS` /
arXiv:2601.20180. The source archive is pinned to SHA-256
`c32199596640624de68ae92f19d4db2324d837580da51db25910213388262b76`.
