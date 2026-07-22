# On the Computational Complexity of Performative Prediction

CPU-only source-pinned finite certificate for ICML 2026 OpenReview
`kkhVljGiMS` / arXiv:2601.20180. It verifies explicit formulas and reduction
controls while leaving PPAD/PLS completeness proofs scoped to the primary source.

## Reproduce

```bash
python3 -m venv .venv
.venv/bin/python repro/src/run_publication_gate.py
```

The only dependency is the Python standard library. The gate verifies the SHA-256
of `source/arxiv-2601.20180.tar`, then validates the following six anchored claims:

1. The explicit epsilon-prime threshold `0.088/6`.
2. Hardness retained under quadratic loss and affine distribution shifts.
3. The stated near-transition tractability regime.
4. The stated exponential ERM-query scale.
5. The well-bounded convex-domain reduction.
6. Strategic local-optimum hardness and a finite one-label local-optimum check.

`outputs/publication_gate.json` is fail-closed: it is emitted only after the
verifier and independent test complete successfully. This package is CPU-only;
it does not use a GPU or Hugging Face compute.

## Scope

The archive is pinned to SHA-256
`c32199596640624de68ae92f19d4db2324d837580da51db25910213388262b76`.
The finite certificates check explicit consequences and guard against two
incorrect alternatives. They do not independently prove PPAD/PLS completeness;
the primary source remains the authority for those complexity proofs.
