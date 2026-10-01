"""Fetch pinned public comparison checkpoints without loading a GPU model."""
import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
locks = json.loads((ROOT/'configs/community-models-lock.json').read_text(encoding='utf-8'))
report = {}
for name, info in locks.items():
    selected = [file for file in info['files'] if file in ['README.md','LICENSE','option_marker.pt']
                or (file.endswith(('.json','.safetensors','.py')) and not file.startswith(('assets/','eval/','multilingual/','typed-decisions/')))]
    print(f'Downloading {name}@{info["revision"]}', flush=True)
    # Local-dir downloads do not require Windows symlink privileges.
    path = snapshot_download(name, revision=info['revision'], allow_patterns=selected,
                             local_dir=str(ROOT/'.cache/community'/name.replace('/', '--')/info['revision']), max_workers=2)
    report[name] = {'revision': info['revision'], 'path': path, 'files': selected}
    print(f'Cached {name}', flush=True)
with (ROOT/'handoff/community-checkpoints.json').open('x', encoding='utf-8') as stream:
    json.dump(report, stream, indent=2)
