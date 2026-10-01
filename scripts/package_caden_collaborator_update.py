"""Build and verify a new private Caden collaborator handoff, preserving v1."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
result = subprocess.check_output([sys.executable, str(ROOT / 'scripts/package_research_snapshot.py')], cwd=ROOT, text=True)
report = json.loads(result)
snapshot = ROOT / report['archive']
target = ROOT / 'exports/caden-arr-collaborator-handoff-2026-10-01-v3.zip'
if target.exists():
    raise FileExistsError('Preserve existing handoff version')
patterns = [re.compile(p) for p in [rb'apikey_[A-Za-z0-9_]{50,}', rb'hf_[A-Za-z0-9]{30,}', rb'sk-[A-Za-z0-9_-]{30,}']]
with zipfile.ZipFile(snapshot) as archive:
    for entry in archive.infolist():
        if any(p.search(archive.read(entry)) for p in patterns):
            raise ValueError('Credential pattern in ' + entry.filename)
    en = archive.read('paper/arr/main_en.tex').decode('utf-8')
    zh = archive.read('paper/arr/中文对应稿.md').decode('utf-8')
    assert 'Caden: Compact Candidate Encoders' in en and 'Caden：' in zh
    for url in ['https://github.com/Minnesinger02/caden', 'https://huggingface.co/Leonard02/caden-encoder-mixed',
                'https://huggingface.co/Leonard02/candidate-qwen3-06b-lora-mixed']:
        assert url in en and url in zh
    publication = json.loads(archive.read('handoff/caden-publication-completed.json'))
    assert publication['verified'] and publication['github_has_issues'] is False
    manifest = json.loads(archive.read('REPLICATION_MANIFEST.json'))
shutil.copyfile(snapshot, target)
assert hashlib.sha256(target.read_bytes()).hexdigest() == report['sha256']
report.update(archive=target.relative_to(ROOT).as_posix(), credential_pattern_scan_passed=True,
    collaborator_draft_identifying_links=True, version=3, published_release=publication,
    checkpoints_separate='exports/jev-local-checkpoints-2026-10-01.zip')
target.with_suffix('.audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
target.with_suffix('.zip.sha256').write_text(report['sha256']+'  '+target.name+'\n',encoding='utf-8')
index_path = ROOT / 'exports/HANDOFF_INDEX.json'
index = json.loads(index_path.read_text(encoding='utf-8'))
index.setdefault('historical_main_packages', []).append(index['main_package'])
index['main_package'] = report
index['english_master_sha256'] = manifest['english_master_sha256']
index['local_checkpoints_package']['current_manuscript_sha256'] = manifest['english_master_sha256']
index['local_checkpoints_package']['unchanged_weights'] = True
index['local_checkpoints_package']['contains_caden_v2'] = False
index['current_caden_v2_weights'] = publication['models'][0]
index['local_checkpoints_package']['note'] = 'Preserved original checkpoint package, excluding v2/pilot weights; its manuscript SHA records v1. Download current v2 weights at the pinned Hugging Face revision.'
index_path.write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
