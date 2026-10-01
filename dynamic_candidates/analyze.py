"""Probability truncation and paired candidate-order diagnostics; no model calls."""
import argparse
import json
from pathlib import Path

from decision_lab.common import metrics, softmax


def topk_renorm(probs, k):
    if k < 1:
        raise ValueError("k must be positive")
    keep = sorted(probs, key=probs.get, reverse=True)[:k]
    mass = sum(probs[key] for key in keep)
    if mass <= 0:
        raise ValueError("No retained probability mass")
    return {key: (p / mass if key in keep else 0.0) for key, p in probs.items()}, mass


def load(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate prediction IDs")
    return rows


def fit_temperature(rows):
    if not rows or any(r.get("split") != "calibration" for r in rows):
        raise ValueError("Use only a separate split=calibration file")
    if any("logits" not in r or "candidate_order" not in r for r in rows):
        raise ValueError("Calibration requires successful decoder logits for every example")
    candidates = [2 ** (i / 8) for i in range(-24, 33)]
    def loss(t):
        import math
        return sum(-math.log(max(softmax(r["logits"], t)[r["candidate_order"].index(r["label"])], 1e-15)) for r in rows) / len(rows)
    return min(candidates, key=loss)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--calibration", help="Optional separate calibration predictions from identical model/prompt configuration")
    p.add_argument("--paired", help="Optional shuffled run on identical IDs and semantic candidate sets")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    rows = load(args.predictions)
    transformed, masses = [], []
    for r in rows:
        if "probabilities" not in r:
            transformed.append(r)
            continue
        probs, mass = topk_renorm(r["probabilities"], args.top_k)
        masses.append(mass)
        transformed.append({**r, "probabilities": probs})
    result = {"original": metrics(rows), "topk_renormalized": metrics(transformed),
              "retained_mass_mean": sum(masses) / len(masses) if masses else None,
              "note": "Dropped candidates retain zero probability; NLL clipped at 1e-15. No samples are dropped."}
    if args.calibration:
        calibration = load(args.calibration)
        groups = lambda records: {r.get("group_id", r["id"]) for r in records}
        if groups(calibration) & groups(rows) or {r["id"] for r in calibration} & {r["id"] for r in rows}:
            raise ValueError("Calibration overlaps evaluation")
        t = fit_temperature(calibration)
        calibrated = []
        for r in rows:
            if "probabilities" not in r:
                calibrated.append(r)
            else:
                calibrated.append({**r, "probabilities": dict(zip(r["candidate_order"], softmax(r["logits"], t)))})
        result.update(temperature=t, calibrated=metrics(calibrated))
    if args.paired:
        other = {r["id"]: r for r in load(args.paired)}
        if set(other) != {r["id"] for r in rows}:
            raise ValueError("Paired runs must have identical IDs")
        flips, distances = [], []
        for r in rows:
            s = other[r["id"]]
            if "probabilities" not in r or "probabilities" not in s:
                continue
            a, b = r["probabilities"], s["probabilities"]
            if set(a) != set(b) or r["label"] != s["label"]:
                raise ValueError("Paired comparison requires same semantic candidates and gold label")
            # Stable semantic tie breaking for order comparison.
            winner = lambda p: max(sorted(p), key=p.get)
            flips.append(int(winner(a) != winner(b)))
            distances.append(sum(abs(a[k] - b[k]) for k in a) / 2)
        result["paired"] = {"valid_pairs": len(flips), "total": len(rows),
                            "flip_rate": sum(flips) / len(flips) if flips else None,
                            "mean_total_variation": sum(distances) / len(distances) if distances else None}
    with Path(args.out).open("x") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
