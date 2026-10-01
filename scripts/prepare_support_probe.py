"""Freeze controlled support changes and correct-option absence on development only."""
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from decision_lab.common import read_data

NONE_KEY = 'none_of_listed_intents'


def perturb(row, condition, vocabulary):
    options = dict(row['criteria'])
    label = row['label']
    if NONE_KEY in vocabulary or NONE_KEY in options:
        raise ValueError('Reserved absence key collides with intent vocabulary')
    if condition == 'drop_nongold':
        key = random.Random('support-drop-v1:' + row['id']).choice([k for k in options if k != label])
        del options[key]
    elif condition == 'add_nongold':
        key = random.Random('support-add-v1:' + row['id']).choice(sorted(set(vocabulary) - set(options)))
        options[key] = vocabulary[key]
    elif condition == 'gold_absent':
        del options[label]
        options[NONE_KEY] = 'None of the listed intents describes the message.'
        label = NONE_KEY
    elif condition != 'original':
        raise ValueError('Unknown support condition')
    return {**row, 'id': row['id'] + '__support_' + condition, 'reference_id': row['id'],
            'condition': condition, 'criteria': options, 'label': label,
            'oracle_shortlist': condition != 'gold_absent'}


def main():
    source = ROOT / 'data/pilot16/dev.jsonl'
    vocabulary_source = ROOT / 'data/pilot16-8792/train.jsonl'
    output = ROOT / 'data/support-dev8'
    if output.exists():
        raise FileExistsError(output)
    vocabulary = {}
    for row in read_data(vocabulary_source):
        for key, text in row['criteria'].items():
            if key in vocabulary and vocabulary[key] != text:
                raise ValueError('Inconsistent intent descriptions')
            vocabulary[key] = text
    data = read_data(source)
    probes = [perturb(row, condition, vocabulary) for row in data
              for condition in ['original', 'drop_nongold', 'add_nongold', 'gold_absent']]
    random.Random(1729).shuffle(probes)
    output.mkdir()
    path = output / 'dev.jsonl'
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in probes), encoding='utf-8')
    manifest = {'items': len(probes), 'source_questions': len(data), 'conditions': {'original': 8, 'drop_nongold': 7, 'add_nongold': 9, 'gold_absent': 8},
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'vocabulary_source_sha256': hashlib.sha256(vocabulary_source.read_bytes()).hexdigest(),
                'scope': 'development diagnostic; removing the gold intent and adding an explicit none option is a separate absence task, not oracle-eight accuracy or CLINC OOS',
                'statistical_unit': '200 source questions, not 800 independent observations',
                'structural_expectation': 'independent encoder surviving raw logits remain invariant when nongold support changes; probabilities need not remain invariant; no such expectation for joint encoder/decoder',
                'selection': 'do not tune a none threshold on these labels; report argmax and failures by condition; absence results are exploratory'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
