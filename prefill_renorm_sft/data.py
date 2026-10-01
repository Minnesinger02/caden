"""Convert exported Jev requests plus labels/answers; never call a remote API."""
import argparse
import json
from pathlib import Path
from decision_lab.common import validate_probs, read_data


def as_text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)


def convert(record):
    if record.get("split") not in {"train", "calibration", "test", "smoke"}:
        raise ValueError("Each record needs a preassigned train/calibration/test/smoke split")
    if not isinstance(record.get("id"), str):
        raise ValueError("Each exported request needs a string id")
    request = record["request"]
    answers = record.get("response", {}).get("answers", {})
    gold = record.get("gold", {})
    result = []
    for qid, question in request["questions"].items():
        if question["type"] != "choice":
            raise ValueError("Pilot supports Choice only; split other types into a separate export")
        criteria = {k: as_text(v) if v is not None else k for k, v in question["criteria"].items()}
        teacher = answers.get(qid, {}).get("probabilities")
        if teacher is not None:
            validate_probs(teacher, criteria)
        label = gold.get(qid)
        source = "gold"
        if label is None:
            if teacher is None:
                raise ValueError("Inputs alone are insufficient: provide gold or response probabilities")
            label = max(teacher, key=teacher.get)
            source = "teacher"
        if label not in criteria:
            raise ValueError("Gold is not a supplied candidate")
        row = {"id": json.dumps([record["id"], qid]), "group_id": record.get("group_id", record["id"]),
               "split": record["split"], "state": as_text(request["state"]),
               "question": as_text(question["instructions"]), "criteria": criteria, "label": label,
               "label_source": source}
        if teacher is not None:
            row["teacher_probabilities"] = teacher
            row["teacher_model"] = record.get("response", {}).get("model", "unspecified")
        result.append(row)
    if not result:
        raise ValueError("No questions in request")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rows = []
    for line in Path(a.input).read_text().splitlines():
        if line.strip():
            rows.extend(convert(json.loads(line)))
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate request/question IDs")
    group_splits = {}
    for row in rows:
        group = row["group_id"]
        if group in group_splits and group_splits[group] != row["split"]:
            raise ValueError("Same document/request group occurs in multiple splits")
        group_splits[group] = row["split"]
    # Validate before creating the requested output.
    import tempfile
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "data.jsonl"
        path.write_text(text)
        read_data(path)
    with Path(a.out).open("x") as f:
        f.write(text)
    print(f"Converted {len(rows)} Choice questions; no API calls")


if __name__ == "__main__":
    main()
