"""Record pinned author declarations separately from exact-text overlap evidence."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    provenance = json.loads((ROOT / 'handoff/release-provenance.json').read_text(encoding='utf-8'))
    entries = {Path(entry['evidence']).name: entry for entry in provenance['entries']}
    cases = [
        ('Von-README.md', 'Von', 'Banking77', 'Training-data table explicitly names BANKING77 customer-intent routing.'),
        ('Decider2B-README.md', 'Decider 2B', 'human-labelled public sets (training halves)', 'Supervised source table names BANKING77 among human-labelled training halves; exact evaluated-item overlap not established here.'),
        ('Nano-README.md', 'OpenDecider Nano', '- clinc/clinc_oos', 'Dataset metadata lists CLINC. Training narrative claims exclusion of its published benchmark families, which is not independently certified for our custom CLINC evaluation.'),
        ('kev08-README.md', 'Kev', '- legacy-datasets/banking77', 'Dataset metadata lists BANKING77; separate local exact-text audit covers one public Kev suite only.'),
    ]
    rows = []
    for filename, model, marker, declaration in cases:
        entry = entries[filename]
        source = ROOT / entry['evidence']
        raw = source.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != entry['sha256']:
            raise ValueError('Pinned model card changed')
        lines = raw.decode('utf-8').splitlines()
        hits = [i+1 for i, line in enumerate(lines) if marker.lower() in line.lower()]
        if not hits:
            raise ValueError(f'Declaration not found: {model}')
        rows.append({'model': model, 'declared_information': declaration, 'line_numbers': hits,
            'source_url': entry['source_url'], 'model_revision': entry['revision'],
            'evidence': entry['evidence'], 'sha256': digest,
            'evidence_type': 'author declaration in pinned card, not independently verified per-item training overlap'})
    report = {'rows': rows,
        'scope': 'Public benchmark leave-out is local, not a certificate of community model novelty. Training declarations and exact-text suite overlap are different evidence types; absence of a declaration is not absence of exposure.',
        'exact_text_suite_audit': 'handoff/kev-full-confirmation-overlap.json',
        'not_established': ['Complete training/model-selection histories for all releases', 'Community CLINC test-item novelty', 'Equivalent upstream training exposure']}
    audit_path = ROOT / report['exact_text_suite_audit']
    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)
    report['exact_text_suite_audit_sha256'] = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    (ROOT / 'handoff/training-exposure-declarations.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = ['# Pinned training-exposure declarations', '', report['scope'], '',
        '| Release | Declared information | Evidence |', '|---|---|---|']
    for row in rows:
        lines.append(f"| {row['model']} | {row['declared_information']} | [{Path(row['evidence']).name}]({row['source_url']}); lines {row['line_numbers']} |")
    lines += ['', 'These declarations do not replace a complete per-item overlap audit. Source file hashes and model revisions are in training-exposure-declarations.json.']
    (ROOT / 'handoff/TRAINING_EXPOSURE.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
