"""Build a portable source bundle; exclude local environments, secrets and weights."""
import hashlib
import json
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
out = root/'dist'
out.mkdir(exist_ok=True)
archive = out/'jev16-windows-handoff-2026-09-30.zip'
roots = ['README.md', 'START_HERE_WINDOWS.md', 'pyproject.toml', 'uv.lock', '.gitignore',
         'decision_lab', 'dynamic_candidates', 'prefill_renorm_sft', 'tests', 'scripts', 'configs', 'handoff', 'data/smoke.jsonl', 'results/uniform-smoke', 'results/topk-smoke.json']
excluded = {'.git', '.venv', '.venv-win', '__pycache__', '.pytest_cache', '.DS_Store', 'node_modules'}
files = {}
for name in roots:
    path = root/name
    paths = path.rglob('*') if path.is_dir() else [path]
    for file in paths:
        rel = file.relative_to(root)
        if not file.is_file() or file.is_symlink() or any(p in excluded for p in rel.parts):
            continue
        if file.name.startswith('.env') or file.suffix in {'.pyc', '.safetensors', '.bin', '.pt', '.zip'}:
            continue
        files[rel.as_posix()] = file.read_bytes()
manifest = {'format': 1, 'snapshot': '2026-09-30', 'files': {k: hashlib.sha256(v).hexdigest() for k,v in sorted(files.items())},
            'excluded': 'Virtualenvs, caches, credentials, downloaded weights/datasets, future results; includes historical smoke/CPU reports only.'}
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for name, data in sorted(files.items()):
        z.writestr('jev16/'+name, data)
    z.writestr('jev16/PACKAGE_MANIFEST.json', json.dumps(manifest, indent=2, ensure_ascii=False))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for name, expected in manifest['files'].items():
        assert hashlib.sha256(z.read('jev16/'+name)).hexdigest() == expected
checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
archive.with_suffix('.zip.sha256').write_text(f'{checksum}  {archive.name}\n', encoding='utf-8')
print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size, 'source_files': len(files), 'sha256': checksum}, indent=2))
