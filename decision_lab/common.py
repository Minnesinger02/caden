import hashlib
import json
import math
import platform
import statistics
from pathlib import Path


def read_data(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    seen = set()
    for r in rows:
        if not isinstance(r.get("id"), str) or r["id"] in seen:
            raise ValueError("Each record needs a unique string id")
        seen.add(r["id"])
        if not isinstance(r.get("state"), str) or not isinstance(r.get("question"), str):
            raise ValueError("state and question must be strings")
        c = r.get("criteria")
        if not isinstance(c, dict) or not 2 <= len(c) <= 255:
            raise ValueError("criteria must contain 2–255 candidates")
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in c.items()):
            raise ValueError("Candidate IDs and descriptions must be strings")
        if r.get("label") not in c:
            raise ValueError("Gold label must be in criteria (use an explicit other label if needed)")
    if not rows:
        raise ValueError("Empty dataset")
    return rows


def softmax(values, temperature=1.0):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    scaled = [x / temperature for x in values]
    high = max(scaled)
    exps = [math.exp(x - high) for x in scaled]
    total = sum(exps)
    return [x / total for x in exps]


def validate_probs(probs, keys, sum_tolerance=1e-5):
    if set(probs) != set(keys):
        raise ValueError("Probability keys differ from requested candidates")
    vals = list(probs.values())
    if any(not isinstance(p, (int, float)) or not math.isfinite(p) or p < 0 or p > 1 for p in vals):
        raise ValueError("Invalid probability")
    if not math.isclose(sum(vals), 1.0, abs_tol=sum_tolerance):
        raise ValueError("Probabilities do not sum to one")


def normalize_rounded_probs(probs, keys, decimals=4):
    """Normalize only sum errors explainable by the known wire rounding."""
    validate_probs(probs, keys, sum_tolerance=len(probs) * .5 * 10 ** -decimals + 1e-8)
    total = sum(probs.values())
    if total <= 0:
        raise ValueError("Rounded probability sum must be positive")
    return {key: value / total for key, value in probs.items()}


def percentile(xs, q):
    xs = sorted(xs)
    pos = (len(xs) - 1) * q
    lo = int(pos)
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (pos - lo)


def metrics(predictions):
    valid = [r for r in predictions if "probabilities" in r]
    if not valid:
        return {"total": len(predictions), "valid": 0, "failures": len(predictions)}
    correct, conf, nll, brier = [], [], [], []
    for r in valid:
        p = r["probabilities"]
        validate_probs(p, p.keys())
        winner = max(p, key=p.get)
        correct.append(int(winner == r["label"]))
        conf.append(p[winner])
        nll.append(-math.log(max(p.get(r["label"], 0.0), 1e-15)))
        brier.append(sum((v - int(k == r["label"])) ** 2 for k, v in p.items()))
    ece = 0.0
    for b in range(10):
        idx = [i for i, c in enumerate(conf) if min(int(c * 10), 9) == b]
        if idx:
            ece += len(idx) / len(valid) * abs(statistics.mean(conf[i] for i in idx) - statistics.mean(correct[i] for i in idx))
    # Group ties: do not claim thresholds that split equally confident predictions.
    risk_curve = []
    for threshold in sorted(set(conf), reverse=True):
        accepted = [i for i, c in enumerate(conf) if c >= threshold]
        risk_curve.append({"threshold": threshold, "coverage_all": len(accepted) / len(predictions),
                           "error_rate": 1 - statistics.mean(correct[i] for i in accepted)})
    latency = [r["latency_ms"] for r in valid if "latency_ms" in r]
    result = {"total": len(predictions), "valid": len(valid), "failures": len(predictions) - len(valid),
              "accuracy_all_failures_wrong": sum(correct) / len(predictions),
              "accuracy_valid": statistics.mean(correct), "nll_clipped_1e-15": statistics.mean(nll),
              "brier_sum": statistics.mean(brier), "ece_10_equal_width": ece,
              "risk_coverage_diagnostic_not_threshold_selection": risk_curve}
    if latency:
        result.update(latency_p50_ms=percentile(latency, .5), latency_p95_ms=percentile(latency, .95))
    return result


def write_run(out, data, predictions, metadata):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in predictions), encoding='utf-8')
    metadata.update(dataset_sha256=hashlib.sha256(Path(data).read_bytes()).hexdigest(),
                    python=platform.python_version(), platform=platform.platform())
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    (out / "metrics.json").write_text(json.dumps(metrics(predictions), indent=2), encoding='utf-8')


def sync(device):
    import torch
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    elif str(device).startswith("mps"):
        torch.mps.synchronize()
