import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class TestCertificate(unittest.TestCase):
    def test_six_source_pinned_claims(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'verification.json'
            subprocess.run(
                [sys.executable, 'repro/src/verify_performative_complexity.py', '--output', str(output)],
                cwd=ROOT,
                check=True,
            )
            result = json.loads(output.read_text())
            self.assertEqual(result['verified_claims'], 6)
            self.assertEqual(result['falsified_claims'], 0)
            self.assertTrue(all(result['negative_controls'].values()))
