"""CPU-only exact tokenizer preflight for all frozen support perturbations."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from transformers import AutoTokenizer
from decision_lab.common import read_data
from prefill_renorm_sft.model import encode
from scripts.encoder_workload import question_workload


def main():
    path = ROOT / 'data/support-dev8/dev.jsonl'
    rows = read_data(path)
    lock = json.loads((ROOT / 'configs/resolved_revisions-parquet.json').read_text(encoding='utf-8'))
    encoder = AutoTokenizer.from_pretrained(ROOT / 'checkpoints/encoder16-s42-e3-8792/backbone', local_files_only=True)
    decoder = AutoTokenizer.from_pretrained(lock['decoder']['id'], revision=lock['decoder']['revision'], local_files_only=True)
    groups = {}
    for row in rows:
        by_condition = groups.setdefault(row['reference_id'], {})
        if row['condition'] in by_condition:
            raise ValueError('Duplicate support probe condition')
        by_condition[row['condition']] = row
    expected_sizes = {'original': 8, 'drop_nongold': 7, 'add_nongold': 9, 'gold_absent': 8}
    for conditions in groups.values():
        if set(conditions) != set(expected_sizes):
            raise ValueError('Incomplete support conditions')
        base = conditions['original']
        for name, row in conditions.items():
            if len(row['criteria']) != expected_sizes[name] or (row['state'], row['question'], row['group_id']) != (base['state'], base['question'], base['group_id']):
                raise ValueError('Support probe changed question identity or count')
    report = {'dataset_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'purpose': 'CPU input validation only; no model scores or timing', 'source_questions': len(groups), 'conditions': {}}
    for condition in expected_sizes:
        subset = [r for r in rows if r['condition'] == condition]
        joint = [question_workload(r, encoder, 'joint')['max_row_tokens'] for r in subset]
        independent = [question_workload(r, encoder, 'independent')['max_row_tokens'] for r in subset]
        causal = [len(encode(r, decoder, 512)[0]) for r in subset]
        if max(joint + independent + causal) > 512:
            raise ValueError('Support perturbation exceeds registered context budget')
        report['conditions'][condition] = {'items': len(subset), 'candidates': expected_sizes[condition],
                                           'max_joint_tokens': max(joint), 'max_independent_row_tokens': max(independent),
                                           'max_qwen_tokens': max(causal)}
    out = ROOT / 'handoff/support-probe-preflight.json'
    with out.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
