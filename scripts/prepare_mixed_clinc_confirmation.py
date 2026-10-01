"""Extract unscored CLINC confirmation rows without changing candidate payloads."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'data/mixed-banking-clinc8/test.jsonl'
    manifest = json.loads((source.parent / 'manifest.json').read_text(encoding='utf-8'))
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest['files']['test']['sha256']:
        raise ValueError('Mixed confirmation source changed')
    lines = source.read_bytes().splitlines(keepends=True)
    selected = [line for line in lines if json.loads(line)['domain'] == 'clinc']
    if len(selected) != manifest['files']['test']['clinc']:
        raise ValueError('CLINC confirmation count mismatch')
    old = {json.loads(line)['group_id'] for line in (ROOT / 'data/clinc-transfer8/test.jsonl').read_text(encoding='utf-8').splitlines() if line}
    rows = [json.loads(line) for line in selected]
    if {r['group_id'] for r in rows} & old or len({r['group_id'] for r in rows}) != len(rows):
        raise ValueError('Fresh confirmation overlaps old scored probe or duplicates groups')
    output = ROOT / 'data/mixed-clinc-confirmation8'
    output.mkdir(exist_ok=False)
    path = output / 'test.jsonl'
    path.write_bytes(b''.join(selected))
    result = {'items': len(rows), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'source_sha256': manifest['files']['test']['sha256'], 'source': str(source.relative_to(ROOT)),
              'payload': 'exact source JSONL bytes for CLINC rows, same candidate dictionaries and labels',
              'task': 'in-scope CLINC oracle eight candidates; not 150-way or OOS',
              'status_when_frozen': 'unscored; old 300-question probe excluded',
              'comparison': 'same fixed fresh questions for all three BANKING-only versus supervised mixed-domain local seeds and released community models; recipes fixed before scores; upstream pretraining exposure not certified'}
    (output / 'manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
