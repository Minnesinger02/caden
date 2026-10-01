"""Bundle all local training checkpoints separately; exclude downloaded Hub caches."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'exports/jev-local-checkpoints-2026-10-01.zip'


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    paths = sorted(p for p in (ROOT / 'checkpoints').rglob('*') if p.is_file())
    if not paths or OUT.exists():
        raise ValueError('Missing checkpoints or existing archive; preserve previous artifact')
    OUT.parent.mkdir(exist_ok=True)
    entries = []
    readme = ('# Local training checkpoint supplement\n\n'
        'Extract into the same root as jev-arr-collaborator-handoff-2026-10-01.zip. '
        'All locally saved checkpoints are included under checkpoints/. '
        'Qwen LoRA adapters require the pinned upstream base model; cached upstream and community weights are excluded. '
        'Original pretrained asset licenses still apply. This is a private research supplement, not an anonymous submission. '
        'Use scripts/verify_handoff_zip.py to check the embedded file manifest.\n').encode()
    with zipfile.ZipFile(OUT, 'x', allowZip64=True) as archive:
        archive.writestr('CHECKPOINT_README.md', readme)
        entries.append({'path': 'CHECKPOINT_README.md', 'bytes': len(readme), 'sha256': hashlib.sha256(readme).hexdigest()})
        for index, path in enumerate(paths):
            if not path.resolve().is_relative_to((ROOT / 'checkpoints').resolve()):
                raise ValueError('Checkpoint resolves outside named directory')
            name = path.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_STORED if path.suffix.lower() in {'.pt', '.pth', '.bin', '.safetensors'} else zipfile.ZIP_DEFLATED
            digest, size = hashlib.sha256(), 0
            with path.open('rb') as source, archive.open(info, 'w', force_zip64=True) as target:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
                    target.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            entries.append({'path': name, 'bytes': size, 'sha256': digest.hexdigest()})
            if index % 40 == 0:
                print(json.dumps({'checkpoints_files_written': index + 1, 'total': len(paths)}), flush=True)
        manifest = {'scope': 'All saved local checkpoint files, including weights/adapters and configs; no Hub cache/environment/API credential.',
            'english_master_sha256': file_hash(ROOT / 'paper/arr/main_en.tex'), 'files': entries}
        archive.writestr('CHECKPOINT_MANIFEST.json', json.dumps(manifest, indent=2))
    # Reading every byte also exercises ZIP CRC checks, then confirms SHA256.
    with zipfile.ZipFile(OUT) as archive:
        for entry in entries:
            digest, size = hashlib.sha256(), 0
            with archive.open(entry['path']) as source:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
                    size += len(chunk)
            if digest.hexdigest() != entry['sha256'] or size != entry['bytes']:
                raise ValueError('Archived checkpoint integrity mismatch')
    report = {'archive': OUT.relative_to(ROOT).as_posix(), 'bytes': OUT.stat().st_size,
        'files': len(entries), 'checkpoint_files': len(paths), 'sha256': file_hash(OUT),
        'all_archived_hashes_verified': True, 'english_master_sha256': manifest['english_master_sha256']}
    OUT.with_suffix('.audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
