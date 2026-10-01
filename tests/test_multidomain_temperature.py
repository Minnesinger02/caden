"""Guard continuation calibration against data leakage and broken logits."""
import unittest
import json
from scripts.train_caden_multidomain_v2 import fit_calibration

class TemperatureGuardTests(unittest.TestCase):
    def rows(self):
        return [{'split':'calibration','label':gold,'logits':{'a':5.0,'b':0.0},'probabilities':{'a':.993307,'b':.006693}}
                for gold in ['a','a','a','b']]

    def test_overconfident_logits_get_lower_calibration_nll(self):
        result=fit_calibration(self.rows())
        self.assertGreater(result['temperature'],1)
        self.assertLess(result['calibration_nll_after'],result['calibration_nll_before'])
        self.assertEqual(result['valid_items'],4)
        self.assertEqual(json.loads(json.dumps(result))['valid_items'],4)

    def test_training_examples_cannot_fit_temperature(self):
        rows=self.rows();rows[0]['split']='train'
        with self.assertRaises(ValueError):fit_calibration(rows)

    def test_invalid_support_is_rejected(self):
        rows=self.rows();rows[0]['label']='missing'
        with self.assertRaises(ValueError):fit_calibration(rows)

    def test_nonfinite_logits_are_rejected(self):
        rows=self.rows();rows[0]['logits']['a']=float('nan')
        with self.assertRaises(ValueError):fit_calibration(rows)

if __name__=='__main__':unittest.main()
