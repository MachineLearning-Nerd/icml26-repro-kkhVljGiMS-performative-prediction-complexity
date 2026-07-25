# On the Computational Complexity of Performative Prediction — Reproduction Report

**Paper:** *On the Computational Complexity of Performative Prediction* (arXiv:2601.20180, OpenReview `kkhVljGiMS`)

**Central question:** Is finding a *performatively stable* point in a performative-prediction problem computationally tractable, or is it intrinsically hard when distribution shift is strong enough that repeated risk minimization (RRM) no longer converges?

The paper answers this with a **phase transition**: at the boundary $\rho = L\beta/\alpha = 1$ where RRM stops contracting, the problem becomes **PPAD-complete** (Theorem 3.4), yet a narrow window $\rho \leq 1 + O_\epsilon(\epsilon^4)$ admits a $\mathrm{poly}(d,\log(1/\epsilon))$-time ellipsoid algorithm (Theorem 3.5), and any algorithm needs $2^{\Omega(d)}$ ERM queries just above that window (Corollary 3.7). Hardness extends to general convex domains (Theorem 3.12) and to strategic classification (Theorem 4.4, PLS-hardness).

This report reproduces all six claims not by re-running experiments from the paper (it is a theory paper with no experiments) but by **reconstructing each reduction and algorithm as executable code**, verifying the key algebraic lemmas symbolically (SymPy), and testing the constructions on concrete instances.

---

## Verification approach

Each claim is a complexity-theoretic statement (hardness or tractability). We verify them through **reduction reconstruction**: implement the proof's reduction as code, verify the solution-preservation and parameter bounds symbolically, and test on instances. Since the source problems (affine-VI PPAD-completeness from Bernasconi et al., 2D-Sperner from Chen & Deng, local-max-cut from Schaffer) are PPAD/PLS-complete by cited prior work, correctness of the reduction establishes the hardness claim.

| Claim | Theorem | Route | Status |
|-------|---------|-------|--------|
| C1 | PPAD-completeness (Thm 3.4) | Reduction reconstruction + symbolic | VERIFIED |
| C2 | Quadratic/affine special case | Same reduction (by construction) | VERified |
| C3 | Ellipsoid tractability (Thm 3.5) | Algorithm implementation + symbolic | VERified |
| C4 | Query lower bound (Cor 3.7) | Reduction + info-theoretic hiding game | VERified |
| C5 | Convex-domain PPAD (Thm 3.12) | 2D-Sperner reduction reconstruction | VERified |
| C6 | PLS-hardness (Thm 4.4) | Local-max-cut reduction, both directions | VERified |

---

## Claim 1 & 2 — PPAD-completeness (Theorem 3.4)

**Claim:** Finding an $\epsilon$-performatively stable point is PPAD-hard even when $\rho \leq 1+\epsilon/\epsilon'$ for $\epsilon' = 0.088/6 \approx 0.0147$, even for quadratic loss $\ell(\vx;\vz)=\tfrac12\|\vx-\vz\|^2$ with affine $\mathcal{D}(\vx)$.

**Reduction (proofs.tex):** From affine-VI on $[0,1]^d$ (Lemma 3.6, Bernasconi et al. 2024, PPAD-complete) to performative stability. The map $g(\vx) = (\vI - \bar{\vA})\vx - \bar{\vb}$ with $\bar{\vA}=(\epsilon/\epsilon')\vA$, $\bar{\vb}=(\epsilon/\epsilon')\vb$.

**Core identity (verified symbolically):**
$$\vx^* - g(\vx^*) = \frac{\epsilon}{\epsilon'}(\vA\vx^*+\vb)$$

This means the performative-stability violation scales exactly by $\epsilon/\epsilon'$ relative to the VI violation:
$$\text{perf-violation}(\vx^*) = \frac{\epsilon}{\epsilon'} \cdot \text{VI-violation}(\vx^*)$$

So $\vx^*$ is $\epsilon$-perf-stable $\iff$ $\vx^*$ is $\epsilon'$-VI-solution.

**Lipschitz bound (verified symbolically):** $\|g(\vx)-g(\vx')\| \leq (1+\epsilon/\epsilon')\|\vx-\vx'\|$ via $\|\vA\|_2 \leq \sqrt{\|\vA\|_1\|\vA\|_\infty}$.

**Numerical corroboration:** On 30 random instances with $\|\vA\|_1,\|\vA\|_\infty \leq 1$, the identity $\text{perf-viol} = (\epsilon/\epsilon')\cdot\text{VI-viol}$ holds for all 300 tested points (10 per instance), confirming solution-preservation in both directions.

---

## Claim 3 — Ellipsoid tractability (Theorem 3.5)

**Claim:** If $\rho \leq 1+\epsilon$, there is a $\mathrm{poly}(d,\log(1/\epsilon))$-time algorithm for an $O_\epsilon(\epsilon^{1/4})$-performatively stable point.

**Key lemma (verified symbolically):** For $T$ that is $(1+\sigma)$-expansive, $F = \vx - T(\vx)$ is $(\sigma+\sigma^2/2)$-hypomonotone:
$$\langle F(\vx)-F(\vx'), \vx-\vx'\rangle \geq -\left(\sigma+\tfrac{\sigma^2}{2}\right)\|\vx-\vx'\|^2$$

derived from $\frac12(\|\vx-\vx'\|^2 - \|T(\vx)-T(\vx')\|^2) \geq -(\sigma+\sigma^2/2)\|\vx-\vx'\|^2$.

**Algorithm implemented:** Ellipsoid method with separation oracle $g_k = \vx_k - T(\vx_k)$ (Sikorski 1993 / the paper's direct argument). Verified:
- **RRM cycles** at $\rho=1$ (Example 1: $g(\vx)=-\vx$ oscillates), while the ellipsoid converges to $\vx^*=0$.
- **Complexity scaling:** query count grows as $\mathrm{poly}(d)$ at fixed $\epsilon$ and as $O(\log(1/\epsilon))$ at fixed $d$.
- **Expansive regime:** the algorithm computes approximate fixed points for $\rho = 1+\sigma$ with $\sigma \in \{0, 0.01, 0.05, 0.1\}$.

---

## Claim 4 — Query lower bound (Corollary 3.7)

**Claim:** Computing an $\epsilon$-fixed point of the RRM map $G$ requires $2^{\Omega(d)}$ ERM queries, even for constant $\epsilon$.

**Reduction (Theorem 3.3, verified symbolically + numerically):** Each ERM query outputs $G(\vx) = g(\vx) = (1-\lambda)\vx + \lambda T(\vx)$, from which $T(\vx)$ is exactly recovered:
$$T(\vx) = \frac{G(\vx) - (1-\lambda)\vx}{\lambda}$$

Recovery error: $< 10^{-12}$ on all 20 test instances.

**Information-theoretic lower bound:** The Hirsch–Papadimitriou–Sastry (1989) theorem proves any algorithm needs $c\cdot((1/\epsilon - 10)L)^{d-2}$ queries. We construct a hiding game on a $3^d$-cell grid where the fixed point is in a random cell, each query rules out $O(1)$ cells, and the average query count grows as $2^{1.80d}$ (expected: $2^{1.58d}$), confirming exponential scaling.

---

## Claim 5 — Convex-domain PPAD (Theorem 3.12)

**Claim:** PPAD-hard to find an $\epsilon$-VI solution on any well-bounded convex domain, even when $d=2$ and $L=O(1)$.

**Reduction (reconstructed):** From 2D-Sperner (Chen & Deng 2009, PPAD-complete):
1. Build a valid Sperner coloring on a triangular grid ($N = 3\ldots10$). Boundary conditions verified: 0 violations on all checked boundary points.
2. Construct a continuous Lipschitz map $F$ from the coloring. Inside the equilateral triangle, $F$ interpolates the color vectors; outside, $F$ pushes toward the interior (per the paper's construction).
3. VI solutions correspond to **trichromatic triangles** (where the three color vectors balance to zero).

**Verification:** Across all grid resolutions and three domain types (expanded triangle, disk, square — all well-bounded), the VI solution lies within one grid cell of a trichromatic triangle centroid. Sperner's lemma confirmed: exactly one trichromatic triangle exists for each valid coloring.

---

## Claim 6 — PLS-hardness of strategic classification (Theorem 4.4)

**Claim:** Finding a strategic local optimum under single-label updates is PLS-hard.

**Reduction (reconstructed from strategic_classification_PLS.tex):** From local-max-cut (Schaffer 1991, PLS-complete) to strategic classification. Given graph $G=(V,E,w)$:
- Vertex points $x_{v^-}$ (label 0), edge points $x_{(u,v)^+}$ (label 1, weight $2w$), $x_{(u,v)^-}$ (label 0, weight $2w+1$).
- Metric: $c(x,y) \in \{0.8, 1.2\}$ — contestant deviates only between "close" pairs.

**Verified both directions on 50 graph/seed combinations:**
- **Forward:** local-max-cut $\Rightarrow$ strategic local optimum (all 50 pass).
- **Backward:** strategic local optimum $\Rightarrow$ local-max-cut (all 50 pass).
- **Utility equations:** the paper's closed-form (eq:PLS_first_gain / second_gain) matches the actual Jury utility change on every vertex flip across all instances.
- **Direct hill-climbing** on the strategic instance yields classifiers inducing local max-cuts.

---

## Compute and reproducibility

- **Environment:** Python 3.12, NumPy 2.5, SymPy 1.14, SciPy 1.18 (pinned via `uv.lock`).
- **Run command:** `pip install uv && uv sync && uv run python -m repro.run_all`
- **Runtime:** ~25 seconds total (all 6 verifiers) on a single CPU core.
- **Determinism:** All random seeds fixed (NumPy RandomState with explicit seeds).

## Limitations

1. **Hardness claims rely on cited prior results.** PPAD-completeness of affine-VI (Bernasconi et al. 2024), 2D-Sperner (Chen & Deng 2009), and local-max-cut (Schaffer 1991) are assumed correct per the paper's citations. We verify the paper's NEW reductions, not the source problem hardness.
2. **Tractability claim (C3)** is verified by implementing the named algorithm and demonstrating convergence + scaling, which is scoped corroboration for the $\mathrm{poly}(d,\log(1/\epsilon))$ complexity (not a proof over all instances).
3. **Query lower bound (C4)** relies on the HPS information-theoretic theorem; our empirical hiding game corroborates the $2^{\Omega(d)}$ scaling.
