"""Freeze full local held-out calibration/test pools before computing their scores."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from decision_lab.common import read_data

source = ROOT / "data/banking77-8-parquet"
output = ROOT / "data/confirmation8-full"
if output.exists():
    raise FileExistsError(output)
training = read_data(ROOT / "data/pilot16-8792/train.jsonl")
development = read_data(ROOT / "data/pilot16/dev.jsonl")
excluded = {r["group_id"] for r in training + development}
splits, manifest = {}, {"task": "BANKING77 oracle eight-candidate choice; gold always shortlisted",
                       "protocol": "Freeze full available deduplicated official test and separate calibration pools before evaluation; no test-based model or prompt selection",
                       "scope": "Held out from this project's local training/development, not a guarantee of unseen data for community checkpoints",
                       "community_contamination": "Audit community training and selection corpora separately; report known overlap and do not label all baselines zero-shot",
                       "source": str(source), "files": {}}
for split in ("calibration", "test"):
    path = source / f"{split}.jsonl"
    rows = read_data(path)
    if any(r["split"] != split for r in rows):
        raise ValueError("Source split annotation differs")
    groups = {r["group_id"] for r in rows}
    if len(groups) != len(rows) or groups & excluded:
        raise ValueError("Source contains duplicate groups or local training/development overlap")
    pilot = read_data(ROOT / "data/pilot16" / f"{split}.jsonl")
    indexed = {r["id"]: r for r in rows}
    if any(indexed.get(r["id"]) != r for r in pilot):
        raise ValueError("Frozen pilot items are not unchanged subsets")
    splits[split] = path.read_bytes()
    manifest["files"][split] = {"items": len(rows), "pilot_subset_items": len(pilot),
                                 "sha256": hashlib.sha256(splits[split]).hexdigest()}
    excluded |= groups
output.mkdir()
for split, content in splits.items():
    (output / f"{split}.jsonl").write_bytes(content)
(output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2))
