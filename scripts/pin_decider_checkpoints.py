"""Resolve feasible official Decider releases and archive their small metadata."""
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError

ROOT = Path(__file__).resolve().parents[1]
output = ROOT / "configs/decider-models-lock.json"
if output.exists():
    raise FileExistsError(output)
report = {}
for model_id in ("Mapika/decider-0.8b", "Mapika/decider-2b"):
    try:
        info = HfApi().model_info(model_id, files_metadata=True)
    except RepositoryNotFoundError:
        if model_id.endswith("0.8b"):
            print(f"Optional release unavailable: {model_id}", flush=True)
            continue
        raise
    revision = info.sha
    destination = ROOT / ".cache/community" / model_id.replace("/", "--") / revision
    files = [f.rfilename for f in info.siblings if f.rfilename.endswith((".json", ".safetensors"))
             or f.rfilename in {"README.md", "LICENSE", "tokenizer.model", "merges.txt", "vocab.json"}]
    small = {}
    for name in files:
        if name.endswith(".safetensors"):
            continue
        path = Path(hf_hub_download(model_id, name, revision=revision, local_dir=destination))
        small[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    sizes = {f.rfilename: f.size for f in info.siblings if f.rfilename.endswith(".safetensors")}
    report[model_id] = {"revision": revision, "files": files, "small_file_sha256": small,
                        "weight_file_sizes": sizes, "weights_downloaded": False}
    print(json.dumps({"model": model_id, "revision": revision, "weight_bytes": sum(sizes.values())}), flush=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
