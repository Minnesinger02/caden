"""Pin and unpack public upstream source archives safely; no code execution."""
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
repos = {'laya': 'NandhaKishorM/laya', 'opendecider': 'manjunathshiva/opendecider',
         'von': 'wfzyx/von', 'decider': 'Mapika/decider', 'openjev': 'lookski/openjev'}
report_path = ROOT/'handoff/community-source-snapshots.json'
report = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {}
def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent':'jev-local-research'}), timeout=60) as stream:
        return stream.read()
for name, repo in repos.items():
    output = ROOT/'external'/f'{name}-upstream'
    if output.exists():
        if name in report and report[name]['path'] == str(output):
            continue
        raise FileExistsError(output)
    metadata = json.loads(fetch(f'https://api.github.com/repos/{repo}'))
    commit = json.loads(fetch(f'https://api.github.com/repos/{repo}/commits/{metadata["default_branch"]}'))['sha']
    content = fetch(f'https://codeload.github.com/{repo}/zip/{commit}')
    archive = zipfile.ZipFile(io.BytesIO(content))
    entries = []
    for entry in archive.infolist():
        parts = Path(entry.filename).parts[1:]
        if not parts or entry.is_dir():
            continue
        target = (output/Path(*parts)).resolve()
        if not target.is_relative_to(output.resolve()):
            raise ValueError('Unsafe archive member')
        entries.append((entry, target))
    for entry, target in entries:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(entry))
    report[name] = {'repo':repo,'commit':commit,'archive_sha256':hashlib.sha256(content).hexdigest(),'path':str(output)}
    print(json.dumps(report[name]), flush=True)
    (ROOT/'handoff/community-source-snapshots.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
