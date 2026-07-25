"""Shim that runs the original source-pinned weak verifier (regression continuity)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    subprocess.run(
        [sys.executable, "repro/src/verify_performative_complexity.py",
         "--output", "outputs/verification.json"],
        cwd=ROOT, check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "repro/tests", "-v"],
        cwd=ROOT, check=True,
    )
    verification = json.loads((ROOT / "outputs/verification.json").read_text())
    assert verification["verified_claims"] == 6
    assert verification["falsified_claims"] == 0
    gate = {
        "paper": "kkhVljGiMS",
        "gate": "passed",
        "verifier": "source-pinned-weak (historical baseline regression)",
        "verified_claims": 6,
        "note": "Weak arithmetic/source-anchor checks only; superseded by rigorous per-claim verifiers on child branches.",
    }
    (ROOT / "outputs/publication_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
