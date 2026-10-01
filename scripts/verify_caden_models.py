"""Verify every published model file against the reviewed release manifest."""
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi

root = Path(__file__).resolve().parents[1]
stage = root / 'release-staging/caden'
manifest = json.loads((stage / 'release-manifest.json').read_text(encoding='utf-8'))
progress = json.loads((stage / 'publication-progress.json').read_text(encoding='utf-8'))
api = HfApi()
assert api.whoami()['name'].lower() == 'leonard02'
counts = {}
for record, folder in zip(progress['models'], ['huggingface-encoder', 'huggingface-qwen-lora']):
    info = api.model_info(record['repo'], revision=record['commit'])
    assert not info.private and info.sha == record['commit']
    files = {e.path: e for e in api.list_repo_tree(record['repo'], revision=record['commit'], recursive=True) if hasattr(e, 'blob_id')}
    expected = {e['path']: e for e in manifest['packages'][folder]['files']}
    assert set(files) - {'.gitattributes'} == set(expected), record['repo']
    for path, entry in expected.items():
        remote = files[path]
        assert remote.size == entry['bytes'], path
        if remote.lfs:
            assert remote.lfs.sha256 == entry['sha256'], path
        else:
            data = (stage / folder / path).read_bytes()
            digest = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            assert digest == remote.blob_id, path
    counts[record['repo']] = len(expected)
assert len(counts) == 2
progress['model_files_verified'] = counts
progress['verified'] = progress.get('github_verified') and progress.get('github_has_issues') is False
assert progress['verified']
for name in ['publication-progress.json', 'publication-completed.json']:
    (stage / name).write_text(json.dumps(progress, indent=2), encoding='utf-8')
manifest['published'] = True
manifest['publication'] = progress
(stage / 'release-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps(progress, indent=2))
