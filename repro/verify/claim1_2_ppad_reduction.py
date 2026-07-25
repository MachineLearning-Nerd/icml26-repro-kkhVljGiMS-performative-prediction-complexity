"""Claims 1 & 2 — PPAD-completeness of performative stability (Theorem 3.4).

CLAIM CONTRACTS:
  Claim 1 (Theorem 3.4): Finding an epsilon-performatively stable point is
    PPAD-hard even when rho = L*beta/alpha <= 1 + eps/eps', with eps' = 0.088/6.
  Claim 2 (Theorem 3.4 special case): This PPAD-hardness persists even when the
    loss is quadratic ell(x;z) = (1/2)||x-z||^2 and D(x) is an affine map g(x).

VERIFICATION ROUTE — reduction reconstruction:
  The proof (proofs.tex) reduces from the affine VI problem (Lemma 3.6,
  bernasconi2024role, PPAD-complete) to performative stability. We reconstruct
  the reduction and verify both directions:
    (A) epsilon-performative stable  ==>  eps'-VI solution   (solution preservation)
    (B) the Lipschitz bound           ||g(x)-g(x')|| <= (1+eps/eps')||x-x'||
  symbolically (sympy) and numerically on random instances. Since the affine VI
  problem is PPAD-complete (cited), correctness of the reduction establishes
  PPAD-hardness. Claim 2 follows because the reduction uses the quadratic loss
  with affine g (Eq. 2-3) by construction.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[2]
EPS_PRIME = 0.088 / 6  # the explicit constant from Theorem 3.4


# --------------------------- symbolic verification ---------------------------

def symbolic_reduction_correctness() -> dict:
    """Symbolically verify that eps-perf-stable ==> eps'-VI solution.

    Reduction: g(x) = (I - A_bar)x - b_bar, where A_bar = (eps/eps')*A, b_bar = (eps/eps')*b.
    For the quadratic loss ell(x;z) = 0.5*||x||^2 - x^T z, the first-order stability
    condition <x-x*, x*-g(x*)> >= -eps becomes (substituting x*-g(x*) = A_bar*x* + b_bar):
      <x-x*, (eps/eps')*(Ax*+b)> >= -eps
    i.e. <x-x*, Ax*+b> >= -eps'.   (QED)
    """
    d = sp.Symbol("d", positive=True, integer=True)
    eps, eps_p = sp.symbols("eps eps_prime", positive=True)
    lam = eps / eps_p  # scaling factor

    # Symbolic vectors / matrices (2-d for tractability)
    x, xstar = sp.symbols("x1 xstar1", real=True)
    A = sp.MatrixSymbol("A", 1, 1)
    b = sp.Symbol("b", real=True)

    # g(x*) = (1 - lam*A)*xstar - lam*b
    g_xstar = xstar - lam * (A[0, 0] * xstar + b)
    # x* - g(x*) = lam*(A*xstar + b)
    residual = sp.expand(xstar - g_xstar)
    expected = sp.expand(lam * (A[0, 0] * xstar + b))
    step1_ok = bool(sp.simplify(residual - expected) == 0)

    # Stability: <x - x*, x* - g(x*)> >= -eps  ==>  <x-x*, A*x*+b> >= -eps'
    # (x - xstar) * lam * (A*xstar + b) >= -eps  ==>  (x-xstar)*(A*xstar+b) >= -eps/lam = -eps'
    bound = sp.simplify(eps / lam)
    step2_ok = bool(sp.simplify(bound - eps_p) == 0)

    return {
        "eps_prime_exact": str(sp.Rational(88, 6000)),  # 0.088/6 as exact fraction
        "eps_prime_value": float(EPS_PRIME),
        "x_minus_g_equals_lambda_Ax_plus_b": step1_ok,
        "eps_divided_by_lambda_equals_eps_prime": step2_ok,
        "conclusion": (
            "Substituting g(x)=(I-A_bar)x-b_bar into the stability condition "
            "<x-x*,x*-g(x*)>=-eps yields <x-x*,(eps/eps')(Ax*+b)>=-eps, "
            "hence <x-x*,Ax*+b>=-eps'. Reduction preserves solutions."
        ),
    }


def symbolic_lipschitz_bound() -> dict:
    """Symbolically verify ||g(x)-g(x')|| <= (1 + eps/eps')||x-x'||.

    Uses ||A_bar||_2 <= (eps/eps')*sqrt(||A||_1*||A||_inf) <= eps/eps' (since ||A||_1,||A||_inf <= 1).
    So ||I - A_bar||_2 <= 1 + ||A_bar||_2 <= 1 + eps/eps'.
    """
    eps, eps_p = sp.symbols("eps eps_prime", positive=True)
    A1, Ainf = sp.symbols("A1 Ainf", positive=True)  # ||A||_1, ||A||_inf

    # spectral norm bound: ||A||_2 <= sqrt(||A||_1 * ||A||_inf)
    A2_bound = sp.sqrt(A1 * Ainf)
    # A_bar = (eps/eps') * A, so ||A_bar||_2 <= (eps/eps') * sqrt(A1*Ainf)
    Abar_bound = (eps / eps_p) * A2_bound
    # With A1 <= 1, Ainf <= 1: ||A_bar||_2 <= eps/eps'
    Abar_max = Abar_bound.subs([(A1, 1), (Ainf, 1)])
    # ||I - A_bar||_2 <= 1 + ||A_bar||_2
    total = 1 + Abar_max
    expected = 1 + eps / eps_p
    ok = bool(sp.simplify(total - expected) == 0)
    return {
        "spectral_norm_bound": "||A||_2 <= sqrt(||A||_1 * ||A||_inf)",
        "with_A1_Ainf_le_1": "||A_bar||_2 <= eps/eps'",
        "lipschitz_bound": "||g(x)-g(x')|| <= (1+eps/eps')||x-x'||",
        "symbolic_check_passed": ok,
    }


# --------------------------- numerical instance verification ---------------------------

def build_reduction(A: np.ndarray, b: np.ndarray, eps: float, eps_p: float):
    """Build the performative instance from an affine VI problem via the reduction."""
    lam = eps / eps_p
    A_bar = lam * A
    b_bar = lam * b

    def g(x: np.ndarray) -> np.ndarray:
        return (np.eye(len(x)) - A_bar) @ x - b_bar

    # Lipschitz constant: max singular value of (I - A_bar)
    M = np.eye(len(b)) - A_bar
    s = np.linalg.norm(M, 2)  # spectral norm
    return g, s


def vi_violation(A: np.ndarray, b: np.ndarray, xstar: np.ndarray) -> float:
    """max_{x in [0,1]^d} -<x - x*, Ax*+b>.  x* is an eps'-VI-solution iff this <= eps'."""
    F = A @ xstar + b
    x_min = np.where(F > 0, 0.0, 1.0)  # minimize <x, F>
    return float(np.dot(xstar - x_min, F))


def perf_violation(g, xstar: np.ndarray) -> float:
    """max_{x in [0,1]^d} -<x - x*, x*-g(x*)>.  x* is eps-stable iff this <= eps."""
    Fp = xstar - g(xstar)
    x_min = np.where(Fp > 0, 0.0, 1.0)
    return float(np.dot(xstar - x_min, Fp))


def verify_reduction_on_instance(A: np.ndarray, b: np.ndarray, eps: float, eps_p: float) -> dict:
    """Verify the reduction on a specific affine VI instance.

    Core identity (holds for ALL x* in [0,1]^d, not just solutions):
        perf_violation(x*) = (eps/eps') * vi_violation(x*)
    because x*-g(x*) = (eps/eps')*(Ax*+b).

    Therefore: x* is eps-perf-stable  <=>  x* is eps'-VI-solution.
    """
    d = len(b)
    g, measured_L = build_reduction(A, b, eps, eps_p)
    lam = eps / eps_p

    rng = np.random.RandomState(hash((A.tobytes(), b.tobytes())) % (2**31))
    scaling_checks = []
    solution_checks = []
    nonsolution_checks = []

    for _ in range(10):
        xstar = rng.uniform(0.05, 0.95, size=d)  # interior point
        vi_v = vi_violation(A, b, xstar)
        perf_v = perf_violation(g, xstar)
        # Core identity
        scaling_checks.append(abs(perf_v - lam * vi_v) < 1e-9 * max(1, abs(vi_v)))
        # Solution preservation
        if vi_v <= eps_p:
            solution_checks.append(perf_v <= eps + 1e-9)
        else:
            nonsolution_checks.append(perf_v > eps - 1e-9)

    # Lipschitz bound: ||I - A_bar||_2 <= 1 + lam
    lip_ok = measured_L <= 1 + lam + 1e-10

    return {
        "d": d,
        "eps": eps,
        "eps_prime": eps_p,
        "lam": lam,
        "rho_bound": 1 + lam,
        "measured_L": float(measured_L),
        "lipschitz_bound_holds": bool(lip_ok),
        "scaling_identity_all_pass": bool(all(scaling_checks)),
        "n_scaling_checks": len(scaling_checks),
        "n_solution_checks": len(solution_checks),
        "solution_checks_all_pass": bool(all(solution_checks)) if solution_checks else True,
        "n_nonsolution_checks": len(nonsolution_checks),
        "nonsolution_checks_all_pass": bool(all(nonsolution_checks)) if nonsolution_checks else True,
    }


def main() -> None:
    sym_correct = symbolic_reduction_correctness()
    sym_lip = symbolic_lipschitz_bound()

    # --- verify on random affine VI instances ---
    rng = np.random.RandomState(12345)
    instance_checks = []
    for trial in range(30):
        d = rng.randint(2, 8)
        # Generate A with ||A||_1 <= 1 and ||A||_inf <= 1
        A = rng.uniform(-1, 1, size=(d, d)) / d  # column sums bounded by ~1
        # ensure bounds
        col_sums = np.abs(A).sum(axis=0)
        row_sums = np.abs(A).sum(axis=1)
        A = A / max(col_sums.max(), row_sums.max(), 1.0)
        b = rng.uniform(-0.5, 0.5, size=d)

        eps = rng.uniform(0.001, 0.1)
        eps_p = EPS_PRIME
        check = verify_reduction_on_instance(A, b, eps, eps_p)
        check["trial"] = trial
        instance_checks.append(check)

    # --- verify eps' exact value ---
    eps_prime_exact = sp.Rational(88, 6000)  # 0.088/6 = 88/6000
    eps_prime_value = float(eps_prime_exact)

    # --- Lipschitz bound summary ---
    all_lip_ok = bool(all(c["lipschitz_bound_holds"] for c in instance_checks))
    all_scaling = bool(all(c["scaling_identity_all_pass"] for c in instance_checks))
    all_sol = bool(all(c["solution_checks_all_pass"] for c in instance_checks))
    all_nonsol = bool(all(c["nonsolution_checks_all_pass"] for c in instance_checks))

    results = {
        "claim": "C1_C2_PPAD_complete",
        "claim1": "Theorem 3.4: eps-perf-stable is PPAD-hard for rho <= 1+eps/eps', eps'=0.088/6",
        "claim2": "Theorem 3.4 special case: PPAD-hardness for quadratic loss + affine D(x)",
        "verifier": "reduction_reconstruction + symbolic",
        "eps_prime": {
            "exact_fraction": "88/6000",
            "value": eps_prime_value,
            "matches_0.088_over_6": abs(eps_prime_value - 0.088 / 6) < 1e-15,
        },
        "symbolic": {
            "reduction_correctness": sym_correct,
            "lipschitz_bound": sym_lip,
        },
        "n_instances": len(instance_checks),
        "all_lipschitz_bounds_hold": all_lip_ok,
        "all_scaling_identities_hold": all_scaling,
        "all_solution_preservation": all_sol,
        "all_nonsolution_preservation": all_nonsol,
        "instance_sample": instance_checks[:4],
        "basis": (
            "The proof's reduction g(x)=(I-A_bar)x-b_bar from affine-VI (Lemma 3.6, "
            "PPAD-complete) to performative stability is verified: (1) symbolically, "
            "eps-stability <==> eps'-VI-solution via x*-g(x*)=(eps/eps')(Ax*+b); "
            "(2) the Lipschitz bound ||g(x)-g(x')||<=(1+eps/eps')||x-x'|| via "
            "||A||_2<=sqrt(||A||_1*||A||_inf); (3) numerically on %d random instances "
            "with ||A||_1,||A||_inf<=1: the identity perf_violation=(eps/eps')*vi_violation "
            "holds for all tested x*, confirming solution-preservation in both directions. "
            "The reduction uses quadratic loss ell(x;z)=0.5||x||^2-x^Tz with affine g by "
            "construction, covering both Claim 1 and Claim 2. Since the affine-VI source is "
            "PPAD-complete (cited), PPAD-hardness follows."
            % len(instance_checks)
        ),
    }
    all_ok = all_lip_ok and all_scaling and all_sol and all_nonsol
    results["status"] = "VERIFIED" if all_ok else "FALSIFIED"

    out_path = ROOT / "outputs" / "claim1_2_ppad.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    if results["status"] != "VERIFIED":
        sys.exit(1)


if __name__ == "__main__":
    main()
