"""Pin and download Kev in its separate environment; do not allocate a GPU model."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

from huggingface_hub import HfApi, snapshot_download
from kev.checkpoint import Checkpoint


def main():
    lock_path = ROOT / "configs/kev-resolved.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    else:
        code = subprocess.check_output(["git", "-c", f"safe.directory={ROOT.as_posix()}/external/kev-upstream",
                                        "-C", str(ROOT / "external/kev-upstream"), "rev-parse", "HEAD"], text=True).strip()
        lock = {"model": "jaredpalmer/kev-0.8b", "revision": HfApi().model_info("jaredpalmer/kev-0.8b").sha,
                "code_commit": code}
        with lock_path.open("x", encoding="utf-8") as stream:
            json.dump(lock, stream, indent=2)
    requested = lock["model"] + "@" + lock["revision"]
    print(f"Downloading {requested}", flush=True)
    checkpoint = Checkpoint(requested)
    meta = checkpoint.meta
    if not meta.base_revision:
        raise ValueError("Kev checkpoint does not pin its base revision")
    print(f"Downloading base {meta.base}@{meta.base_revision}", flush=True)
    snapshot_download(meta.base, revision=meta.base_revision, max_workers=2,
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
    report = {**lock, "requested": requested, "base": meta.base, "base_revision": meta.base_revision,
              "temperature": meta.temperature, "lora_rank": meta.lora,
              "head_sha256": hashlib.sha256(Path(checkpoint.file("head.pt")).read_bytes()).hexdigest(),
              "checkpoint_path": str(checkpoint.path), "base_cached": True}
    destination = ROOT / "handoff/kev-checkpoint.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
