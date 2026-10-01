"""Download BANKING77 and create deterministic, disjoint train/calibration/test splits."""
import argparse
import hashlib
import json
import random
from pathlib import Path


def main():
    from datasets import load_dataset
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/banking77")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset", default="legacy-datasets/banking77", help="Parquet mirror compatible with datasets 4.x")
    p.add_argument("--revision", required=True, help="Pin the Hugging Face dataset revision")
    p.add_argument("--candidates", type=int, default=77, help="<77 is an oracle-included shortlist, not retrieval")
    args = p.parse_args()
    if not 2 <= args.candidates <= 77:
        p.error("candidates must be 2–77")
    out = Path(args.out)
    if out.exists():
        raise ValueError("Output already exists; preserve it and choose a new directory")
    ds = load_dataset(args.dataset, revision=args.revision)
    names = ds["train"].features["label"].names
    rng = random.Random(args.seed)
    # Calibration is carved out of training only. Deduplicate text across splits;
    # official test takes priority so no training record duplicates a test state.
    identity = lambda text: hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()
    seen = {identity(r["text"]) for r in ds["test"]}
    groups = {i: [] for i in range(len(names))}
    for i, row in enumerate(ds["train"]):
        h = identity(row["text"])
        if h not in seen:
            groups[row["label"]].append((i, row, h))
            seen.add(h)
    splits = {"train": [], "calibration": [], "test": []}
    for group in groups.values():
        rng.shuffle(group)
        cut = max(1, round(len(group) * .1))
        splits["calibration"].extend(group[:cut])
        splits["train"].extend(group[cut:])
    seen_test = set()
    for i, row in enumerate(ds["test"]):
        h = identity(row["text"])
        if h not in seen_test:
            splits["test"].append((i, row, h))
            seen_test.add(h)
    counts = {}
    out.mkdir(parents=True, exist_ok=False)
    for split, records in splits.items():
        lines = []
        for i, row, h in records:
            gold = row["label"]
            others = [j for j in range(len(names)) if j != gold]
            local_rng = random.Random(str(args.seed) + h)
            selected = [gold] + local_rng.sample(others, args.candidates - 1)
            local_rng.shuffle(selected)
            criteria = {names[j]: names[j].replace("_", " ") for j in selected}
            lines.append(json.dumps({"id": f"banking77-{split}-{i}", "group_id": h, "split": split,
                                    "state": row["text"], "question": "Which banking intent best describes the customer's message?",
                                    "criteria": criteria, "label": names[gold],
                                    "oracle_shortlist": args.candidates < 77}))
        (out / f"{split}.jsonl").write_text("\n".join(lines) + "\n")
        counts[split] = len(lines)
    (out / "manifest.json").write_text(json.dumps({**vars(args), "counts": counts,
        "note": "Candidate subsets contain gold by construction. They are not real retrieval results."}, indent=2))


if __name__ == "__main__":
    main()
