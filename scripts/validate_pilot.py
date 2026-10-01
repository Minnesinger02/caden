"""Preflight input lengths and split identity before any training or score selection."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
from transformers import AutoTokenizer
from decision_lab.common import read_data
from decision_lab.encoder import MARKER
from prefill_renorm_sft.model import encode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='data/pilot16')
    parser.add_argument('--out', default='handoff/data-preflight.json')
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/single_16gb.json").read_text(encoding="utf-8"))
    lock = json.loads((ROOT / config["lock_file"]).read_text(encoding="utf-8"))
    encoder = AutoTokenizer.from_pretrained(lock["encoder"]["id"], revision=lock["encoder"]["revision"], local_files_only=True)
    encoder.add_special_tokens({"additional_special_tokens": [MARKER]})
    decoder = AutoTokenizer.from_pretrained(lock["decoder"]["id"], revision=lock["decoder"]["revision"], local_files_only=True)
    seen = set()
    report = {"purpose": "length/identity preflight only; no test accuracy or model selection", "splits": {}}
    for split in ("train", "dev", "calibration", "test"):
        path = ROOT / args.data_dir / f"{split}.jsonl"
        rows = read_data(path)
        groups = {r["group_id"] for r in rows}
        if seen & groups or len(groups) != len(rows):
            raise ValueError("Group overlap/duplicates")
        seen |= groups
        elengths, dlengths = [], []
        for row in rows:
            if row["split"] != split or len(row["criteria"]) != config["candidates"]:
                raise ValueError("Split or candidate count mismatch")
            text = "\n".join(["State: " + row["state"], "Question: " + row["question"],
                              *[f"Candidate {k}: {v} {MARKER}" for k, v in row["criteria"].items()]])
            length = len(encoder.encode(text))
            if length > config["max_length"]:
                raise ValueError(f"Encoder overlength: {row['id']} length={length}")
            elengths.append(length)
            ids, _ = encode(row, decoder, config["max_length"])
            dlengths.append(len(ids))
        report["splits"][split] = {"count": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                   "encoder_max_tokens": max(elengths), "decoder_max_tokens": max(dlengths),
                                   "encoder_mean_tokens": sum(elengths)/len(rows), "decoder_mean_tokens": sum(dlengths)/len(rows)}
    output = ROOT / args.out
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
