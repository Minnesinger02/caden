"""JSON input validation shared by training and inference."""
import json
from pathlib import Path


def validate_row(row, training=False):
    if not isinstance(row, dict):
        raise ValueError("Each input must be a JSON object")
    if not all(isinstance(row.get(key), str) for key in ("state", "question")):
        raise ValueError("state and question must be strings")
    criteria = row.get("criteria")
    if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255:
        raise ValueError("criteria must contain 2–255 candidates")
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in criteria.items()):
        raise ValueError("Candidate IDs and descriptions must be strings")
    if training:
        if row.get("split") != "train":
            raise ValueError("Training requires explicit split=train on every record")
        if not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError("Training records require nonempty string IDs")
        if row.get("label") not in criteria:
            raise ValueError("Training label must be a candidate ID")
    return row


def read_jsonl(path, training=False):
    rows = []
    seen = set()
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = validate_row(json.loads(line), training)
                if training:
                    if row["id"] in seen:
                        raise ValueError("Duplicate training ID")
                    seen.add(row["id"])
                rows.append(row)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
    if not rows:
        raise ValueError("Empty input file")
    return rows
