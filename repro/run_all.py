"""Entrypoint that runs every claim verifier and aggregates results.

This is the fixed run command executed on every experiment node:
``uv run python -m repro.run_all``

Each verifier prints a JSON result block delimited by BEGIN/END markers so the
run log carries machine-readable evidence (see orx-evidence skill).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_module(module: str) -> dict:
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-m", module],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    elapsed = round(time.time() - t0, 3)
    print(f"===== {module} (exit={proc.returncode}, {elapsed}s) =====")
    print(proc.stdout)
    if proc.stderr:
        print("--- stderr ---", file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
    if proc.returncode != 0:
        print(f"!! {module} FAILED (exit {proc.returncode})", file=sys.stderr)
    return {"module": module, "exit": proc.returncode, "elapsed_s": elapsed}


def main() -> None:
    results = []
    # Existing weak source-pinned verifier (kept for regression continuity).
    results.append(_run_module("repro.src.run_publication_gate_shim"))
    # Rigorous per-claim verifiers (added by child branches; absent on baseline).
    for mod in [
        "repro.verify.claim1_2_ppad_reduction",
        "repro.verify.claim3_ellipsoid",
        "repro.verify.claim4_query_lower",
        "repro.verify.claim5_convex_domain",
        "repro.verify.claim6_pls_reduction",
    ]:
        try:
            import importlib

            importlib.import_module(mod)
        except ModuleNotFoundError:
            continue
        results.append(_run_module(mod))
    failed = [r for r in results if r["exit"] != 0]
    summary = {
        "paper": "kkhVljGiMS",
        "total_verifiers": len(results),
        "failed_verifiers": len(failed),
        "all_passed": len(failed) == 0,
        "results": results,
    }
    (ROOT / "outputs" / "run_all_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print("===== RUN_ALL SUMMARY =====")
    print(json.dumps(summary, indent=2))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
