"""Same-question fresh CLINC comparison; incomplete seed arms stay pending."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from decision_lab.common import read_data, validate_probs
from scripts.paired_bootstrap import read_runs, paired_interval


def audit_metrics(data, predictions):
    indexed = {r['id']: r for r in predictions}
    if len(indexed) != len(predictions) or set(indexed) != {r['id'] for r in data}:
        raise ValueError('Confirmation trace must exactly cover frozen IDs')
    correct, nll, brier, confidence = [], [], [], []
    for row in data:
        pred = indexed[row['id']]
        if (pred['label'], pred['group_id']) != (row['label'], row['group_id']):
            raise ValueError('Confirmation label/group mismatch')
        if 'error' in pred:
            if 'probabilities' in pred:
                raise ValueError('Ambiguous confirmation failure')
            continue
        p = pred['probabilities']
        validate_probs(p, row['criteria'])
        winner = max(p, key=p.get)
        correct.append(int(winner == row['label']))
        confidence.append(p[winner])
        nll.append(-math.log(max(p[row['label']], 1e-15)))
        brier.append(sum((v - int(k == row['label'])) ** 2 for k, v in p.items()))
    ece = 0.
    for bin_id in range(10):
        selected = [i for i, c in enumerate(confidence) if min(int(c * 10), 9) == bin_id]
        if selected:
            ece += len(selected) / len(correct) * abs(statistics.mean(confidence[i] for i in selected) - statistics.mean(correct[i] for i in selected))
    return {'total': len(data), 'valid': len(correct), 'failures': len(data) - len(correct),
            'accuracy_all_failures_wrong': sum(correct) / len(data),
            'nll_clipped_1e-15': statistics.mean(nll) if nll else None,
            'brier_sum': statistics.mean(brier) if brier else None,
            'ece_10_equal_width': ece if correct else None}


def main():
    source = ROOT / 'data/mixed-clinc-confirmation8/test.jsonl'
    manifest = json.loads((source.parent / 'manifest.json').read_text(encoding='utf-8'))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != manifest['sha256']:
        raise ValueError('Fresh CLINC input changed')
    data = read_data(source)
    expected = sorted((r['id'], r['label'], r['group_id']) for r in data)
    report = {'dataset_sha256': digest, 'items': len(data), 'rows': [], 'paired_comparisons': [], 'pending': [],
              'scope': 'same fresh oracle-eight CLINC items; supervised mixed expansion versus BANKING-only local training; calibration sources differ; community upstream exposure not certified'}
    arrays = {}

    def arm(name, folders, seeds):
        if not all((folder / 'metrics.json').exists() for folder in folders):
            report['pending'].append(name)
            return
        scores, identities, groups, data_hash = read_runs(folders)
        if identities != expected or data_hash != digest:
            raise ValueError('Run differs from fresh confirmation identity')
        checked = []
        for index, folder in enumerate(folders):
            metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
            if seeds:
                stage, family, variant = name.split('-')
                training = metadata.get('training', metadata.get('adapter_training', {}))
                expected_training = ROOT / ('data/pilot16-8792/train.jsonl' if stage == 'bankingonly' else 'data/mixed-banking-clinc8/train.jsonl')
                if training.get('seed') != seeds[index] or (ROOT / training['data']).resolve() != expected_training.resolve():
                    raise ValueError('Local confirmation seed or training pool differs from registration')
                if training.get('data_sha256', training.get('dataset_sha256')) != hashlib.sha256(expected_training.read_bytes()).hexdigest():
                    raise ValueError('Local training hash differs from frozen source')
                if metadata.get('dtype') != 'float32' or training['epochs'] != (3 if family == 'encoder' else 1):
                    raise ValueError('Local precision or epoch recipe changed')
                if family == 'encoder' and (training.get('readout') != 'scalar' or training.get('input_mode', 'joint') != 'joint'):
                    raise ValueError('Primary fresh encoder comparison requires joint scalar readout')
                if family == 'qwen' and (training.get('objective') != 'candidate_ce' or training.get('rank') != 8):
                    raise ValueError('Primary fresh Qwen comparison requires candidate CE rank-8 LoRA')
                if variant == 'calibrated':
                    pool = ROOT / ('data/confirmation8-full/calibration.jsonl' if stage == 'bankingonly' else 'data/mixed-banking-clinc8/calibration.jsonl')
                    calibration = metadata['temperature_calibration']
                    if calibration['calibration_dataset_sha256'] != hashlib.sha256(pool.read_bytes()).hexdigest() or calibration['score_source'] != 'raw_logits':
                        raise ValueError('Local calibration source differs from declared protocol')
            predictions = [json.loads(line) for line in (folder / 'predictions.jsonl').read_text(encoding='utf-8').splitlines() if line]
            recomputed = audit_metrics(data, predictions)
            saved = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
            for key, value in recomputed.items():
                if value is not None and not math.isclose(saved[key], value, rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError('Aggregate differs from complete audited trace')
            checked.append(recomputed)
        accuracies = scores.mean(axis=1).tolist()
        report['rows'].append({'arm': name, 'runs': [str(folder.relative_to(ROOT)) for folder in folders], 'seeds': seeds,
             'accuracy_by_seed_or_release': accuracies, 'accuracy_mean': statistics.mean(accuracies),
             'accuracy_seed_sd': statistics.stdev(accuracies) if len(seeds) > 1 else None,
             'nll_mean': statistics.mean(m['nll_clipped_1e-15'] for m in checked) if all(m['valid'] for m in checked) else None,
             'ece_mean': statistics.mean(m['ece_10_equal_width'] for m in checked) if all(m['valid'] for m in checked) else None,
             'failures_total': sum(m['failures'] for m in checked)})
        arrays[name] = (scores, groups)

    for stage in ['bankingonly', 'mixed']:
        for family in ['encoder', 'qwen']:
            for variant in ['raw', 'calibrated']:
                name = f'{stage}-{family}-{variant}'
                folders = [ROOT / f'results/freshclinc-{stage}-{family}-s{seed}-test-{variant}' for seed in [42, 43, 44]]
                arm(name, folders, [42, 43, 44])
            raw, calibrated = f'{stage}-{family}-raw', f'{stage}-{family}-calibrated'
            if raw in arrays and calibrated in arrays and not (arrays[raw][0] == arrays[calibrated][0]).all():
                raise ValueError('Temperature calibration changed choices')
    for model in ['laya', 'nano', 'openjev', 'von', 'decider08', 'decider2b', 'kev']:
        arm('community-' + model, [ROOT / f'results/freshclinc-community-{model}-test-raw'], [])
    pairs = [('mixed-encoder-raw', 'bankingonly-encoder-raw'), ('mixed-qwen-raw', 'bankingonly-qwen-raw'),
             ('mixed-encoder-raw', 'mixed-qwen-raw'), ('bankingonly-encoder-raw', 'bankingonly-qwen-raw')]
    for candidate, reference in pairs:
        if candidate not in arrays or reference not in arrays:
            continue
        delta, interval, count = paired_interval(arrays[candidate][0], arrays[reference][0], arrays[candidate][1])
        report['paired_comparisons'].append({'candidate': candidate, 'reference': reference, 'delta': delta, 'ci95': interval,
                  'groups': count, 'resamples': 5000, 'seed': 0, 'interpretation': 'conditional on three observed seeds; paired material groups, not universal architecture superiority'})
    (ROOT / 'results/fresh-clinc-summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = ['# Fresh CLINC confirmation', '', report['scope'], '', '| Arm | Accuracy | Seed SD | NLL | ECE | Failures |', '|---|---:|---:|---:|---:|---:|']
    for row in report['rows']:
        sd = f"{100*row['accuracy_seed_sd']:.2f} pp" if row['accuracy_seed_sd'] is not None else '—'
        nll = f"{row['nll_mean']:.4f}" if row['nll_mean'] is not None else 'unavailable'
        ece = f"{row['ece_mean']:.4f}" if row['ece_mean'] is not None else 'unavailable'
        lines.append(f"| {row['arm']} | {100*row['accuracy_mean']:.2f}% | {sd} | {nll} | {ece} | {row['failures_total']} |")
    lines += ['', f"Pending complete arms: {len(report['pending'])}; no partial-seed aggregate inserted."]
    (ROOT / 'results/fresh-clinc-summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
