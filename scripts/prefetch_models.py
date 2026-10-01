"""Download only pilot model assets, respecting the pinned Hub commits."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

from huggingface_hub import snapshot_download


def main():
    config = json.loads((ROOT / "configs/single_16gb.json").read_text(encoding="utf-8"))
    lock = json.loads((ROOT / config.get("lock_file", "configs/resolved_revisions.json")).read_text(encoding="utf-8"))
    for key in ("encoder", "decoder"):
        model = lock[key]
        print(f"Downloading {model['id']}@{model['revision']}", flush=True)
        folder = snapshot_download(model["id"], revision=model["revision"], max_workers=2,
                                   allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
        print(f"Cached {key}: {folder}", flush=True)


if __name__ == "__main__":
    main()
