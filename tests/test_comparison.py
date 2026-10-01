import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from decision_lab.common import write_run


class ComparisonTests(unittest.TestCase):
    def test_same_ids_different_questions_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prediction = {"id": "1", "label": "a", "probabilities": {"a": .8, "b": .2}}
            runs = []
            for i, state in enumerate(("first question", "different question")):
                data = root / f"data{i}.jsonl"
                data.write_text(json.dumps({"id": "1", "state": state}), encoding="utf-8")
                out = root / f"run{i}"
                write_run(out, data, [prediction], {})
                runs.append(str(out))
            command = [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/compare_runs.py"),
                       *runs, "--out", str(root / "comparison.csv")]
            rejected = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("Mismatched data SHA256", rejected.stderr)
            (root / "run1/metadata.json").write_text((root / "run0/metadata.json").read_text(encoding="utf-8"), encoding="utf-8")
            accepted = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
