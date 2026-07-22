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
