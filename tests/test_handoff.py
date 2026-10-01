import io
import json
import unittest
from unittest.mock import patch
from decision_lab.__main__ import Jev

class HandoffTests(unittest.TestCase):
    def test_kev_rounding_is_normalized_and_raw_retained(self):
        row = {'state': 'x', 'question': 'q', 'criteria': {'a': 'one', 'b': 'two', 'c': 'three'}}
        raw = {'a': .3333, 'b': .3333, 'c': .3333}
        response = {'answers': {'decision': {'probabilities': raw}}}
        with patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(response).encode())):
            client = Jev('kev', 'http://127.0.0.1:8009/v1/systemone', None, probability_decimals=4)
            self.assertAlmostEqual(sum(client(row).values()), 1)
            self.assertEqual(client.last['response_probabilities_raw'], raw)
        bad = {'answers': {'decision': {'probabilities': {'a': .3, 'b': .3, 'c': .3}}}}
        with patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(bad).encode())):
            with self.assertRaises(ValueError):
                Jev('kev', 'http://127.0.0.1:8009/v1/systemone', None, probability_decimals=4)(row)

    def test_kev_contract_without_jev_secret(self):
        row = {'state': 'hello', 'question': 'choose', 'criteria': {'a': 'one', 'b': 'two'}}
        response = {'model': 'kev-test', 'answers': {'decision': {'probabilities': {'a': .25, 'b': .75}}}}
        with patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(response).encode())) as send:
            client = Jev('kev-latest', 'http://127.0.0.1:8009/v1/systemone', None)
            self.assertEqual(client(row), response['answers']['decision']['probabilities'])
            req = send.call_args.args[0]
            self.assertNotIn('Authorization', req.headers)
            self.assertEqual(json.loads(req.data)['questions']['decision']['criteria'], row['criteria'])
            self.assertEqual(client.last['served_model'], 'kev-test')

    def test_remote_probability_support_must_match(self):
        row = {'state': 'x', 'question': 'q', 'criteria': {'a': 'one', 'b': 'two'}}
        response = {'answers': {'decision': {'probabilities': {'a': 1}}}}
        with patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(response).encode())):
            with self.assertRaises(ValueError):
                Jev('kev', 'http://127.0.0.1:8009/v1/systemone', None)(row)
