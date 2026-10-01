"""Verify the handoff snapshot before modifying files."""
import hashlib
import json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
manifest = json.loads((root/'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
for name, expected in manifest['files'].items():
    path = root/name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f'Missing or changed: {name}')
print(f"Verified {len(manifest['files'])} files")
