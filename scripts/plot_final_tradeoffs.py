"""Measured domain accuracies and separate paired real-task timing intervals."""
import hashlib
import json
import os
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    paths = ['results/fresh-clinc-summary.json', 'results/confirmation-summary.json',
        'results/mixed-banking-test-retention.json', 'results/decider2b-supplement-summary.json',
        'results/replicated-task-fp32-8792/summary-mixed-reference.json']
    fresh, bank, retention, supplement, timing = [json.loads((ROOT / path).read_text(encoding='utf-8')) for path in paths]
    if fresh['pending'] or len(timing['blocks']) != 5 or len(timing['rows']) != 14:
        raise ValueError('All frozen accuracy and repeated timing arms required')
    f = {row['arm']: row for row in fresh['rows']}
    r = {row['family']: row for row in retention['rows'] if row['variant'] == 'raw'}
    b = {row['model']: row for row in bank['community_rows'] if row['variant'] == 'raw'}
    ratios = {row['compared']: row for row in timing['paired_ratios']}
    systems = [('encoder-mixed-eager', 'Mixed encoder'), ('qwen-mixed', 'Mixed Qwen'),
        ('laya', 'Laya'), ('nano', 'Nano'), ('openjev', 'OpenJev'), ('von', 'Von'),
        ('decider08', 'Decider 0.8B'), ('decider2b', 'Decider 2B'), ('kev', 'Kev')]
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 5.6))
    figure.subplots_adjust(left=.145, right=.965, top=.83, bottom=.19, wspace=.35)
    values = []
    for i, (name, label) in enumerate(systems):
        y = len(systems)-1-i
        color = '#0072B2' if name == 'encoder-mixed-eager' else '#D55E00' if name == 'qwen-mixed' else '#555555'
        if name in ['encoder-mixed-eager', 'qwen-mixed']:
            family = 'encoder' if name.startswith('encoder') else 'qwen'
            scores = r[family]['mixed_accuracy_by_seed']
            bank_accuracy, bank_sd = statistics.mean(scores), statistics.stdev(scores)
            clinc = f[f'mixed-{family}-raw']
        else:
            bank_accuracy = (next(row['accuracy_all_failures_wrong'] for row in supplement['rows']
                if row['task'] == 'test' and row['variant'] == 'raw') if name == 'decider2b' else b[name]['accuracy'])
            bank_sd = None
            clinc = f['community-' + name]
        for ax, mean, sd in [(axes[0], bank_accuracy, bank_sd), (axes[1], clinc['accuracy_mean'], clinc['accuracy_seed_sd'])]:
            ax.errorbar(100*mean, y, xerr=100*sd if sd is not None else None,
                fmt='o', color=color, capsize=3, markersize=5)
            ax.annotate(f'{100*mean:.2f}', (100*mean, y), xytext=(-7, 3),
                textcoords='offset points', ha='right', fontsize=8, color=color)
        ratio = ratios[name] if name != 'encoder-mixed-eager' else {'geometric_mean_speed_ratio': 1.0, 'ratio_ci95': [1.0, 1.0]}
        mean = ratio['geometric_mean_speed_ratio']
        lo, hi = ratio['ratio_ci95']
        axes[2].errorbar(mean, y, xerr=np.array([[mean-lo], [hi-mean]]),
            fmt='o', color=color, capsize=3, markersize=5)
        axes[2].annotate(f'{mean:.2f}', (mean, y), xytext=(7, 3), textcoords='offset points', ha='left', fontsize=8, color=color)
        values.append({'model': name, 'bank_accuracy': bank_accuracy, 'bank_seed_sd': bank_sd,
            'clinc_accuracy': clinc['accuracy_mean'], 'clinc_seed_sd': clinc['accuracy_seed_sd'], **ratio})
    labels = [label for _, label in reversed(systems)]
    axes[0].set(yticks=range(len(systems)), yticklabels=labels, xlim=(65, 101), xlabel='Accuracy (%)', title='BANKING: 3,079 questions')
    axes[1].set(xlim=(82, 101), xlabel='Accuracy (%)', title='Fresh CLINC: 4,197 questions')
    axes[2].set(xscale='log', xlim=(.7, 70), xlabel='Mixed encoder / system rate', title='Task rate: five paired blocks')
    axes[2].axvline(1, color='#AAAAAA', linestyle=':', linewidth=1)
    axes[2].set_xticks([1, 5, 10, 30], labels=['1', '5', '10', '30'])
    for i, ax in enumerate(axes):
        ax.set_ylim(-.7, len(systems)-.3)
        if i:
            ax.set_yticks(range(len(systems)), labels=['']*len(systems))
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', alpha=.15)
        ax.set_axisbelow(True)
    figure.suptitle('Accuracy pools and timing workload are separate measurements', fontsize=12, y=.94)
    figure.text(.5, .055,
        'Accuracy error bars: local three-seed sample SD; each community system uses one checkpoint.\n'
        'Timing: identical 200 BANKING development payloads, FP32/B1/two threads; individual paired-block 95% intervals.\n'
        'Mixed BANKING retention and the 2B BANKING supplement are exploratory; upstream exposure and runtime kernels differ.',
        ha='center', fontsize=8)
    output = ROOT / 'paper/figures'
    figure.savefig(output / 'domain_accuracy_runtime.png', dpi=180)
    figure.savefig(output / 'domain_accuracy_runtime.pdf')
    plt.close(figure)
    (output / 'domain_accuracy_runtime-evidence.json').write_text(json.dumps({
        'sources': {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths},
        'rows': values, 'scope': 'Separate accuracy pools and timing workload; seed SD is not confidence, runtime ratios use observed paired blocks; no cross-pool per-item Pareto or architecture causality claim.'}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
