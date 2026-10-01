"""Read credential from a pipe, list models, then make ONE pinned choice call."""
import hashlib
import getpass
import argparse
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='results/jev-authorized-probe')
    args = parser.parse_args()
    out = ROOT / args.out
    if out.exists():
        raise FileExistsError('Preserve probe; do not repeat automatically')
    print('Awaiting credential on stdin; it will not be printed or saved.', flush=True)
    key = getpass.getpass('API credential (hidden): ').strip()
    if not key or any(char.isspace() for char in key):
        raise ValueError('Invalid credential format')
    out.mkdir()
    headers = {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}
    def request(path, body=None):
        req = urllib.request.Request('https://api.typesafe.ai' + path,
            data=json.dumps(body).encode() if body is not None else None, headers=headers)
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=45) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            report = {'http_status': error.code, 'endpoint': path, 'no_retry': True,
                      'scope': 'No response body or credential stored; stop on API error.'}
            (out / 'error.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps(report), flush=True)
            raise SystemExit(1)
        except (urllib.error.URLError, TimeoutError) as error:
            report = {'error': 'connection_or_timeout', 'endpoint': path, 'no_retry': True,
                      'charge_unknown_if_post': body is not None,
                      'reason': str(getattr(error, 'reason', type(error).__name__)).replace(key, '[REDACTED]')}
            (out / 'error.json').write_text(json.dumps(report), encoding='utf-8')
            print(json.dumps(report), flush=True)
            raise SystemExit(1)
        # Do not retain request headers. Redact any unexpected echo of the credential.
        result = json.loads(json.dumps(result).replace(key, '[REDACTED]'))
        return result, time.perf_counter() - started
    models, _ = request('/v1/models')
    (out / 'models.json').write_text(json.dumps(models, indent=2), encoding='utf-8')
    data = ROOT / 'data/confirmation8-full/test.jsonl'
    row = json.loads(data.read_text(encoding='utf-8').splitlines()[0])
    body = {'state': row['state'], 'model': 'jev-1.13.0',
            'questions': {'decision': {'type': 'choice', 'instructions': row['question'], 'criteria': row['criteria']}}}
    result, latency = request('/v1/systemone', body)
    record = {'row': row, 'source_dataset_sha256': hashlib.sha256(data.read_bytes()).hexdigest(),
              'requested_model': body['model'], 'response': result, 'latency_seconds': latency,
              'request_sha256': hashlib.sha256(json.dumps(body).encode()).hexdigest(), 'retries': 0}
    (out / 'choice-probe.json').write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')
    usage = result.get('usage', {})
    print(json.dumps({'models': [item.get('name') for item in models.get('models', [])],
        'served_model': result.get('model'), 'usage': usage, 'latency_seconds': latency,
        'estimated_usd_at_documented_rate': usage.get('input_tokens', 0) * .042 / 1_000_000,
        'response_saved': 'results/jev-authorized-probe/choice-probe.json'}), flush=True)


if __name__ == '__main__':
    main()
