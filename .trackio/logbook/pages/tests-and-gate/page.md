# Tests and gate


---
<!-- trackio-cell
{"type": "code", "id": "cell_4feeac63bb6a", "created_at": "2026-07-22T13:08:37+00:00", "title": "Run publication gate", "command": [".venv/bin/python", "repro/src/run_publication_gate.py"], "exit_code": 0, "duration_s": 0.198}
-->
````bash
$ .venv/bin/python repro/src/run_publication_gate.py
````

exit 0 · 0.2s


````python title=run_publication_gate.py
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

subprocess.run(
    [sys.executable, 'repro/src/verify_performative_complexity.py', '--output', 'outputs/verification.json'],
    cwd=ROOT,
    check=True,
)
subprocess.run(
    [sys.executable, '-m', 'unittest', 'discover', '-s', 'repro/tests', '-v'],
    cwd=ROOT,
    check=True,
)
verification = json.loads((ROOT / 'outputs/verification.json').read_text())
assert verification['verified_claims'] == 6
assert verification['falsified_claims'] == 0
assert all(verification['negative_controls'].values())
gate = {
    'paper': 'kkhVljGiMS',
    'gate': 'passed',
    'tests_passed': True,
    'publication_gate_passed': True,
    'verified_claims': 6,
    'scope': verification['scope'],
}
(ROOT / 'outputs/publication_gate.json').write_text(json.dumps(gate, indent=2) + '\n')
print(json.dumps(gate, indent=2))

````


````output
{
  "paper": "kkhVljGiMS",
  "source_sha256": "c32199596640624de68ae92f19d4db2324d837580da51db25910213388262b76",
  "scope": "Source-pinned complexity-theorem contract plus finite threshold, affine/quadratic, query-scale, convex-domain, and strategic-objective controls; not an independent PPAD/PLS proof.",
  "negative_controls": {
    "wrong_epsilon_denominator_rejected": true,
    "perturbed_affine_fixed_point_rejected": true
  },
  "claims": {
    "C1": {
      "status": "verified",
      "epsilon_prime": 0.014666666666666666,
      "threshold_cells": 4
    },
    "C2": {
      "status": "verified",
      "affine_quadratic_cells": 4
    },
    "C3": {
      "status": "verified",
      "tractability_cells": 4
    },
    "C4": {
      "status": "verified",
      "query_lower_scale_cells": 12
    },
    "C5": {
      "status": "verified",
      "convex_domain_source_anchor": true
    },
    "C6": {
      "status": "verified",
      "strategic_local_control": true
    }
  },
  "verified_claims": 6,
  "falsified_claims": 0
}
test_six_source_pinned_claims (test_certificate.TestCertificate.test_six_source_pinned_claims) ... {
  "paper": "kkhVljGiMS",
  "source_sha256": "c32199596640624de68ae92f19d4db2324d837580da51db25910213388262b76",
  "scope": "Source-pinned complexity-theorem contract plus finite threshold, affine/quadratic, query-scale, convex-domain, and strategic-objective controls; not an independent PPAD/PLS proof.",
  "negative_controls": {
    "wrong_epsilon_denominator_rejected": true,
    "perturbed_affine_fixed_point_rejected": true
  },
  "claims": {
    "C1": {
      "status": "verified",
      "epsilon_prime": 0.014666666666666666,
      "threshold_cells": 4
    },
    "C2": {
      "status": "verified",
      "affine_quadratic_cells": 4
    },
    "C3": {
      "status": "verified",
      "tractability_cells": 4
    },
    "C4": {
      "status": "verified",
      "query_lower_scale_cells": 12
    },
    "C5": {
      "status": "verified",
      "convex_domain_source_anchor": true
    },
    "C6": {
      "status": "verified",
      "strategic_local_control": true
    }
  },
  "verified_claims": 6,
  "falsified_claims": 0
}
ok

----------------------------------------------------------------------
Ran 1 test in 0.048s

OK
{
  "paper": "kkhVljGiMS",
  "gate": "passed",
  "tests_passed": true,
  "publication_gate_passed": true,
  "verified_claims": 6,
  "scope": "Source-pinned complexity-theorem contract plus finite threshold, affine/quadratic, query-scale, convex-domain, and strategic-objective controls; not an independent PPAD/PLS proof."
}

````
