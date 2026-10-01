"""Same-checkpoint numerical agreement and deployment speed ablation."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out', required=True)
args = parser.parse_args()
prefix = 'encoder16-s42-e3-8792'
baseline = ROOT/'results'/f'{prefix}-dev'
def read(path):
    return (json.loads((path/'metadata.json').read_text(encoding='utf-8')),
            {r['id']:r for line in (path/'predictions.jsonl').read_text(encoding='utf-8').split('\n')
             if line.strip() for r in [json.loads(line)]},
            json.loads((path/'metrics.json').read_text(encoding='utf-8')))
meta, reference, baseline_metrics = read(baseline)
report = []
for name, dtype, attention in [('fp32-eager','float32','eager'),('fp32-sdpa','float32','sdpa'),
                               ('bf16-eager','bfloat16','eager'),('bf16-sdpa','bfloat16','sdpa')]:
    folder = baseline if name=='fp32-eager' else ROOT/'results'/f'{prefix}-{name}-dev'
    other_meta, predictions, metrics = read(folder)
    assert other_meta['dataset_sha256'] == meta['dataset_sha256']
    assert other_meta['checkpoint'] == meta['checkpoint'] and set(predictions)==set(reference)
    disagreements, max_error = [], 0.
    for key, row in reference.items():
        other = predictions[key]
        assert row['label']==other['label'] and row['group_id']==other['group_id']
        assert 'error' not in row and 'error' not in other
        p, q = row['probabilities'], other['probabilities']
        assert set(p)==set(q)
        max_error = max(max_error, max(abs(p[c]-q[c]) for c in p))
        if max(p,key=p.get)!=max(q,key=q.get):
            disagreements.append(key)
    benchmark = json.loads((ROOT/'results'/f'fixed-compute-encoder-8792-{name}.json').read_text(encoding='utf-8'))
    assert benchmark['dtype']==dtype and benchmark['attention']==attention
    report.append({'name':name, 'dtype':dtype, 'attention':attention,
                   'accuracy':metrics['accuracy_all_failures_wrong'],'nll':metrics['nll_clipped_1e-15'],
                   'latency_p50_ms':metrics['latency_p50_ms'],'latency_p95_ms':metrics['latency_p95_ms'],
                   'max_probability_abs_error_vs_fp32_eager':max_error,
                   'argmax_disagreement_count':len(disagreements),'argmax_disagreement_ids':disagreements,
                   'fixed_compute':benchmark['rows']})
output = {'checkpoint':meta['checkpoint'],'dataset_sha256':meta['dataset_sha256'], 'rows':report,
          'interpretation':'implementation/precision ablation on identical trained weights; numerical argmax drift is not an accuracy-training improvement'}
with Path(args.out).open('x', encoding='utf-8') as stream:
    json.dump(output,stream,indent=2)
print(json.dumps([{k:v for k,v in row.items() if k not in ['fixed_compute','argmax_disagreement_ids']} for row in report],indent=2))
