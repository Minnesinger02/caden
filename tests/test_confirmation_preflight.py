"""Confirm the entire queue is fresh before loading its first GPU model."""
import tempfile
import unittest
from pathlib import Path

from scripts.run_mixed_confirmation import confirmation_paths, validate_artifact_paths


class ConfirmationPreflightTests(unittest.TestCase):
    def populate(self, root):
        required, fresh = confirmation_paths()
        for path in required:
            target = root / path
            if path.suffix == '.json':
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text('{}', encoding='utf-8')
            else:
                target.mkdir(parents=True)
        return required, fresh

    def test_complete_prerequisites_and_fresh_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, fresh = self.populate(root)
            self.assertEqual(len(fresh), len(set(fresh)))
            validate_artifact_paths(root)

    def test_existing_last_stage_blocks_whole_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, fresh = self.populate(root)
            target = root / fresh[-1]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('preserved evidence', encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, 'existing='):
                validate_artifact_paths(root)
            self.assertEqual(target.read_text(encoding='utf-8'), 'preserved evidence')

    def test_missing_final_checkpoint_blocks_whole_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            required, _ = self.populate(root)
            (root / required[-1]).rmdir()
            with self.assertRaisesRegex(RuntimeError, 'checkpoints.*s44-trainmixed23789'):
                validate_artifact_paths(root)


if __name__ == '__main__':
    unittest.main()
