"""Record the specifically reviewed fresh-confirmation Chinese snapshot."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper/arr'
sys.path.insert(0, str(ROOT))
from scripts.build_chinese_translation import SECTIONS, tables


def main():
    source = (OUT / 'main_en.tex').read_text(encoding='utf-8-sig')
    abstract = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', source, re.S).group(1).strip()
    matches = list(re.finditer(r'\\section\*?\{([^}]+)\}', source))
    sections = {}
    for i, match in enumerate(matches):
        end = matches[i+1].start() if i+1 < len(matches) else source.index(r'\begin{thebibliography}')
        sections[match.group(1)] = source[match.end():end].strip()
    if set(sections) != set(SECTIONS):
        raise ValueError('Section translations incomplete')
    hashes = {key: hashlib.sha256(value.encode()).hexdigest() for key, value in {'abstract': abstract, **sections}.items()}
    lock_path = OUT / 'translation-source-lock.json'
    original = json.loads(lock_path.read_text(encoding='utf-8'))
    changed = {key for key, value in hashes.items() if original['section_sha256'].get(key) != value}
    reviewed = {'abstract', 'Supervised Domain Expansion and Fresh CLINC Confirmation', 'Discussion', 'Conclusion', 'Limitations'}
    if changed != reviewed:
        raise ValueError(f'Unexpected changes outside this reviewed snapshot: {changed}')
    backup = OUT / 'translation-source-lock-before-fresh.json'
    with backup.open('x', encoding='utf-8') as stream:
        json.dump(original, stream, indent=2)
    lock_path.write_text(json.dumps({'section_sha256': hashes,
        'scope': '2026-10-01 fresh CLINC and exploratory BANKING retention: renewed section-level Chinese review by local research agent; no external peer review',
        'reviewed_changed_sections': sorted(reviewed)}, indent=2), encoding='utf-8')
    subprocess.run([sys.executable, str(ROOT / 'scripts/build_chinese_translation.py')], cwd=ROOT, check=True)
    chinese = (OUT / '中文对应稿.md').read_text(encoding='utf-8')
    projected = [tables(body) for body in sections.values() if tables(body)]
    if any(table not in chinese for table in projected):
        raise ValueError('Chinese table differs from projected master')
    report = {'english_master_sha256': hashlib.sha256((OUT / 'main_en.tex').read_bytes()).hexdigest(),
        'reviewed_changed_sections': sorted(reviewed), 'sections': len(sections), 'table_count': len(projected),
        'projected_table_lines_verified': sum(len([line for line in table.splitlines() if line.startswith('|')]) for table in projected),
        'scope': 'Tables projected exactly from English. Changed narratives reviewed against English and real audited experiment results; no external language review, PDF rendering or submission-readiness claim.'}
    (OUT / 'translation-audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
