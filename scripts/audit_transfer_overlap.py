"""Audit frozen transfer items against explicitly named local reference splits."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from decision_lab.common import read_data


def normalized(text):
    return " ".join(text.casefold().split())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer", default="data/clinc-transfer8/test.jsonl")
    parser.add_argument("--reference-dir", default="data/pilot16-8792")
    parser.add_argument("--splits", nargs="+", choices=["train", "dev", "calibration", "test"], default=["train", "dev", "calibration", "test"])
    parser.add_argument("--out", default="handoff/clinc-full-training-overlap.json")
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        parser.error("Refusing to overwrite an audit")
    transfer = read_data(args.transfer)
    report = {"normalization": "Unicode casefold and whitespace collapse; exact normalized text only",
              "transfer_file": args.transfer, "transfer_sha256": hashlib.sha256(Path(args.transfer).read_bytes()).hexdigest(),
              "transfer_items": len(transfer), "references": {},
              "limitation": "Does not detect semantic paraphrases or overlap with community training corpora."}
    for split in args.splits:
        path = Path(args.reference_dir) / f"{split}.jsonl"
        references = read_data(path)
        texts = {}
        for row in references:
            texts.setdefault(normalized(row["state"]), []).append(row["id"])
        overlaps = [{"transfer_id": row["id"], "reference_ids": texts[normalized(row["state"])]}
                    for row in transfer if normalized(row["state"]) in texts]
        report["references"][split] = {"file": str(path), "items": len(references),
                                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                      "overlap_items": len(overlaps), "overlaps": overlaps}
    report["no_exact_local_overlap"] = all(r["overlap_items"] == 0 for r in report["references"].values())
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"no_exact_local_overlap": report["no_exact_local_overlap"],
                      "counts": {k: v["overlap_items"] for k, v in report["references"].items()}}, indent=2))


if __name__ == "__main__":
    main()
