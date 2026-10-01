"""Saved sequential service throughput; never relabel probabilities as tokens."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from transformers import AutoTokenizer
from decision_lab.common import read_data
from decision_lab.encoder import MARKER
from scripts.encoder_workload import question_workload

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--runs', nargs='+', required=True)
parser.add_argument('--out', required=True)
parser.add_argument('--data', default='data/pilot16/dev.jsonl')
args = parser.parse_args()
data = ROOT / args.data
items = {r['id']: r for r in read_data(data)}
digest = hashlib.sha256(data.read_bytes()).hexdigest()
report = []
for run in args.runs:
    folder = ROOT / run
    metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    assert metadata['dataset_sha256'] == digest
    predictions = [json.loads(line) for line in (folder / 'predictions.jsonl').read_text(encoding='utf-8').split('\n') if line.strip()]
    assert set(r['id'] for r in predictions) == set(items) and len(predictions) == len(items)
    token_counts = [r.get('input_tokens') for r in predictions]
    workloads = None
    if metadata.get('backend') == 'encoder':
        tokenizer = AutoTokenizer.from_pretrained(Path(metadata['checkpoint']) / 'backbone', local_files_only=True)
        mode = metadata.get('training', {}).get('input_mode', 'joint')
        workloads = [question_workload(items[p['id']], tokenizer, mode) for p in predictions]
        token_counts = [w['valid_tokens'] for w in workloads]
    assert all(t is not None for t in token_counts), 'Missing input length'
    elapsed = sum(r['latency_ms'] for r in predictions) / 1000
    report.append({'run': run, 'items': len(predictions), 'total_timed_seconds': elapsed,
                   'decisions_per_second': len(predictions) / elapsed,
                   'input_tokens_per_second': sum(token_counts) / elapsed,
                   'mean_input_tokens': sum(token_counts) / len(predictions), 'generated_tokens_per_second': None,
                   'failures': sum('error' in r for r in predictions), 'scope': metadata['latency_scope']})
    if workloads:
        report[-1].update(encoder_input_mode=mode,
                          mean_encoder_rows=sum(w['encoder_rows'] for w in workloads) / len(workloads),
                          mean_padded_input_tokens=sum(w['padded_tokens'] for w in workloads) / len(workloads),
                          token_scope='sum of nonpadding tokens across actual encoder rows; independent mode counts repeated premise; padded tokens recorded separately')
output = {'dataset_sha256': digest, 'gpu': 'RTX 4090 Laptop GPU', 'dtype': 'float32', 'batch_size': 1,
          'rows': report, 'limitation': 'Rate = item count / sum of synchronized saved item latencies, excludes inter-item bookkeeping. Different tokenizers/prompts; token rates not equal semantic workload. Probability scores are not generated tokens.'}
with Path(args.out).open('x', encoding='utf-8') as stream:
    json.dump(output, stream, indent=2)
print(json.dumps(report, indent=2))
