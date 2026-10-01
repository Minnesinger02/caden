"""Pinned, no-retry Jev comparison; credential only in memory, conservative meter."""
import concurrent.futures
import getpass
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from decision_lab.common import read_data, validate_probs, write_run

PRICE = .042 / 1_000_000
CAP = .50
RESERVE = 64000 * PRICE
MODEL = 'jev-1.13.0'
WORKERS = 4


class Meter:
    def __init__(self, initial, requests=1):
        self.spent = initial
        self.reserved = 0.
        self.requests = requests
        self.unknown = 0
        self.lock = threading.Lock()

    def start(self):
        with self.lock:
            if self.spent + self.reserved + RESERVE > CAP:
                raise RuntimeError('Conservative budget gate reached')
            self.reserved += RESERVE
            self.requests += 1

    def finish(self, input_tokens):
        with self.lock:
            self.reserved -= RESERVE
            if not isinstance(input_tokens, int) or not 0 <= input_tokens <= 64000:
                self.spent += RESERVE
                self.unknown += 1
                raise RuntimeError('Unknown usage; stop and retain worst-case reserve')
            self.spent += input_tokens * PRICE


def normalize_wire(raw, criteria):
    # Infer only the observable decimal grid, never infer unavailable true logits.
    decimals = next((digits for digits in range(2, 7) if all(
        math.isclose(value * 10 ** digits, round(value * 10 ** digits), abs_tol=1e-7)
        for value in raw.values())), None)
    tolerance = len(raw) * .5 * 10 ** (-decimals) + 1e-8 if decimals else 1e-5
    validate_probs(raw, criteria, sum_tolerance=tolerance)
    total = sum(raw.values())
    return {label: value / total for label, value in raw.items()}, decimals


def main():
    base = ROOT / 'results/jev113-budgeted-v2'
    if base.exists():
        raise FileExistsError('Preserve current run; no automatic retry/resume')
    probe = json.loads((ROOT / 'results/jev-authorized-probe-network/choice-probe.json').read_text(encoding='utf-8'))
    if probe['response']['model'] != MODEL:
        raise ValueError('Pinned probe model mismatch')
    inputs = [('banking', 'data/confirmation8-full/test.jsonl'),
              ('fresh-clinc', 'data/mixed-clinc-confirmation8/test.jsonl'),
              ('policy', 'data/policy-transfer8/test.jsonl'),
              ('banking-dev', 'data/pilot16-8792/dev.jsonl')]
    datasets = [(name, path, read_data(ROOT / path)) for name, path in inputs]
    if datasets[0][2][0] != probe['row']:
        raise ValueError('Probe row differs from frozen source')
    previous_root = ROOT / 'results/jev113-budgeted'
    previous = json.loads((previous_root / 'stopped.json').read_text(encoding='utf-8'))
    previous_rows = [json.loads(line) for line in (previous_root / 'banking-responses.jsonl').read_text(encoding='utf-8').splitlines()]
    cached_rows = {row['id']: row for row in previous_rows}
    if len(cached_rows) != len(previous_rows) or previous['unknown_usage_requests']:
        raise ValueError('Prior records duplicate or have unknown usage; review budget first')
    meter = Meter(previous['estimated_spent_usd'], previous['attempted_paid_requests_including_probe'])
    base.mkdir()
    plan = {'model': MODEL, 'concurrency': WORKERS, 'user_balance_usd': 5, 'run_cap_estimated_usd': CAP,
            'input_price_per_million_usd': .042, 'output_price_usd': 0,
            'pricing_source': 'https://docs.typesafe.ai/models', 'checked_date': '2026-10-01',
            'reserve_per_inflight_request_usd': RESERVE, 'retries': 0,
            'probe_reused': True, 'prior_valid_responses_reused': len(cached_rows),
            'prior_paid_requests': previous['attempted_paid_requests_including_probe'],
            'manual_recovery_scope': 'Previously validated results reused without requests. Four unpersisted paid responses from the stopped attempt are explicitly remeasured once; automatic retries remain disabled.',
            'datasets': [{'name': name, 'path': path, 'items': len(rows),
                'sha256': hashlib.sha256((ROOT / path).read_bytes()).hexdigest()} for name, path, rows in datasets],
            'scope': 'Later fixed-version API comparator on previously scored frozen tasks; no model/data/seed selection from Jev results. API latency includes network/server and concurrency4, not local GPU timing. Costs estimated at documented price, not an account billing statement.'}
    (base / 'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    key = getpass.getpass('API credential (hidden; memory only): ').strip()
    if not key:
        raise ValueError('Empty credential')
    stop = threading.Event()
    raw_root = base / 'paid-response-cache'
    raw_root.mkdir()

    def predict(row, cached=None):
        item = {'id': row['id'], 'label': row['label'], 'split': row.get('split'),
                'group_id': row.get('group_id', row['id'])}
        if stop.is_set():
            raise RuntimeError('Stopped after earlier error')
        if cached is None:
            meter.start()
            payload = {'state': row['state'], 'model': MODEL, 'questions': {'decision':
                {'type': 'choice', 'instructions': row['question'], 'criteria': row['criteria']}}}
            request = urllib.request.Request('https://api.typesafe.ai/v1/systemone',
                data=json.dumps(payload).encode(), headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=45) as response:
                    result = json.load(response)
                elapsed = time.perf_counter() - started
            except Exception as error:
                meter.finish(None)
                stop.set()
                raise RuntimeError('API request failed; no retry; charge conservatively reserved') from None
            try:
                meter.finish(result.get('usage', {}).get('input_tokens'))
            except Exception:
                stop.set()
                raise
            safe_result = json.loads(json.dumps(result).replace(key, '[REDACTED]'))
            cache_name = hashlib.sha256(row['id'].encode()).hexdigest() + '.json'
            (raw_root / cache_name).write_text(json.dumps({'id': row['id'], 'response': safe_result,
                'latency_seconds': elapsed}, ensure_ascii=False), encoding='utf-8')
        else:
            result, elapsed = cached['response'], cached['latency_seconds']
        try:
            if result.get('model') != MODEL:
                raise ValueError('Served model changed')
            answer = result['answers']['decision']
            raw = answer['probabilities']
            probabilities, wire_decimals = normalize_wire(raw, row['criteria'])
            total = sum(raw.values())
            validate_probs(probabilities, row['criteria'])
            if answer['choice'] not in raw:
                raise ValueError('Official choice outside candidate support')
            item.update(probabilities=probabilities, response_probabilities_raw=raw,
                        response_probability_sum=total, official_choice=answer['choice'],
                        confidence=answer.get('confidence'), served_model=result['model'],
                        usage=result['usage'], latency_ms=elapsed * 1000, reused_probe=cached is not None,
                        observed_wire_decimal_grid=wire_decimals,
                        official_choice_is_probability_maximum=(raw[answer['choice']] == max(raw.values())))
            return item
        except Exception as error:
            stop.set()
            raise RuntimeError('Response validation failed: ' + str(error).replace(key, '[REDACTED]')) from None

    def save_meter():
        with meter.lock:
            report = {'estimated_spent_usd': meter.spent, 'reserved_inflight_usd': meter.reserved,
                      'attempted_paid_requests_including_probe': meter.requests, 'unknown_usage_requests': meter.unknown,
                      'cap_estimated_usd': CAP, 'pricing_source': plan['pricing_source']}
        (base / 'budget.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        return report

    started = time.perf_counter()
    completed = []
    try:
        for name, path, rows in datasets:
            predictions = []
            journal = base / (name + '-responses.jsonl')
            with journal.open('x', encoding='utf-8') as stream, concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
                remaining = iter(rows)
                pending = {}
                if name == 'banking':
                    first = next(remaining)
                    saved = predict(first, probe)
                    predictions.append(saved)
                    stream.write(json.dumps(saved, ensure_ascii=False) + '\n')
                    stream.flush()
                def submit():
                    try:
                        row = next(remaining)
                    except StopIteration:
                        return False
                    cached = None
                    if name == 'banking' and row['id'] in cached_rows:
                        old = cached_rows[row['id']]
                        if old['label'] != row['label'] or old['group_id'] != row['group_id'] or old['served_model'] != MODEL:
                            raise ValueError('Saved response identity mismatch')
                        cached = {'response': {'model': MODEL, 'usage': old['usage'], 'answers': {'decision':
                            {'probabilities': old['response_probabilities_raw'], 'choice': old['official_choice'], 'confidence': old.get('confidence')}}},
                            'latency_seconds': old['latency_ms'] / 1000}
                    pending[pool.submit(predict, row, cached)] = row['id']
                    return True
                for _ in range(WORKERS):
                    submit()
                while pending:
                    done, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in done:
                        pending.pop(future)
                        saved = future.result()
                        predictions.append(saved)
                        stream.write(json.dumps(saved, ensure_ascii=False) + '\n')
                        stream.flush()
                        if len(predictions) % 100 == 0:
                            print(json.dumps({'dataset': name, 'completed': len(predictions), 'total': len(rows), **save_meter()}), flush=True)
                        submit()
            by_id = {row['id']: row for row in predictions}
            ordered = [by_id[row['id']] for row in rows]
            metadata = {'backend': 'jev-api', 'model': MODEL, 'requested_model': MODEL,
                'concurrency': WORKERS, 'retries': 0, 'warmup': 0, 'precision': 'server_unknown',
                'latency_scope': 'HTTP transport plus server, concurrency4; probe single-concurrency latency reused for first BANK row; no local GPU speed attribution',
                'probability_processing': 'Retain wire values; infer observable decimal grid2..6 and bound sum rounding error by K*0.5*10^-digits, otherwise tolerance1e-5. Normalize sum; official-choice disagreements retained. Probability argmax defines accuracy. No true-logit NLL inferred.',
                'plan_sha256': hashlib.sha256((base / 'plan.json').read_bytes()).hexdigest(),
                'classification': 'late supplementary fixed-version comparator on previously scored tasks'}
            write_run(base / name, ROOT / path, ordered, metadata)
            completed.append(name)
            print(json.dumps({'dataset_completed': name, 'items': len(rows), **save_meter()}), flush=True)
    except Exception as error:
        stop.set()
        report = {'error_type': type(error).__name__, 'stop_reason': str(error).replace(key, '[REDACTED]'),
                  'completed_datasets': completed, **save_meter()}
        (base / 'stopped.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report), flush=True)
        raise SystemExit(1)
    report = {'completed_datasets': completed, 'wall_seconds': time.perf_counter() - started, **save_meter()}
    (base / 'completed.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
