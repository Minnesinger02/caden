"""Plot audited three-seed fresh CLINC expansion and calibration results."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    source = ROOT / 'results/fresh-clinc-summary.json'
    report = json.loads(source.read_text(encoding='utf-8'))
    rows = {row['arm']: row for row in report['rows']}
    if report['items'] != 4197:
        raise ValueError('Expected frozen 4197-question confirmation')
    families = [('encoder', 'Encoder', '#0072B2'), ('qwen', 'Qwen', '#D55E00')]
    for family, _, _ in families:
        for stage in ['bankingonly', 'mixed']:
            for variant in ['raw', 'calibrated']:
                row = rows[f'{stage}-{family}-{variant}']
                if row['seeds'] != [42, 43, 44] or row['failures_total']:
                    raise ValueError('Require complete three-seed, zero-failure local arms')
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.1))
    fig.subplots_adjust(left=.065, right=.98, bottom=.24, top=.80, wspace=.43)
    for family, label, color in families:
        points = [rows[f'{stage}-{family}-raw'] for stage in ['bankingonly', 'mixed']]
        axes[0].errorbar([0, 1], [100*r['accuracy_mean'] for r in points],
            yerr=[100*r['accuracy_seed_sd'] for r in points], fmt='o-',
            color=color, label=label, capsize=4, linewidth=1.8)
        for x, row in enumerate(points):
            axes[0].annotate(f"{100*row['accuracy_mean']:.2f}", (x, 100*row['accuracy_mean']),
                xytext=(0, -15 if family == 'encoder' else 9), textcoords='offset points',
                ha='center', fontsize=9, color=color)
    axes[0].set(xticks=[0, 1], xticklabels=['BANK-only\n8,792 train', 'Mixed\n23,789 train'],
        xlim=(-.25, 1.25), ylim=(75, 101), ylabel='Accuracy (%)', title='Three seeds: mean ± SD')
    axes[0].legend(frameon=False, loc='lower right', fontsize=9)
    evidence = []
    for i, (family, label, color) in enumerate(families):
        comparison = next(r for r in report['paired_comparisons']
            if r['candidate'] == f'mixed-{family}-raw' and r['reference'] == f'bankingonly-{family}-raw')
        delta = 100*comparison['delta']
        lo, hi = [100*v for v in comparison['ci95']]
        axes[1].errorbar(delta, 1-i, xerr=np.array([[delta-lo], [hi-delta]]),
            fmt='o', color=color, capsize=5, markersize=7)
        axes[1].text(.6 if family == 'qwen' else delta, 1-i+.19,
            f'{delta:+.2f} [{lo:.2f}, {hi:.2f}]', ha='left' if family == 'qwen' else 'center', color=color, fontsize=9)
        evidence.append(comparison)
    axes[1].set(yticks=[1, 0], yticklabels=['Encoder', 'Qwen'], ylim=(-.5, 1.5),
        xlim=(0, 22), xlabel='Mixed minus BANK-only (pp)', title='Paired 95% group intervals')
    for family, label, color in families:
        points = [rows[f'mixed-{family}-{variant}'] for variant in ['raw', 'calibrated']]
        axes[2].plot([0, 1], [r['nll_mean'] for r in points], 'o-', color=color, label=label)
        for x, row in enumerate(points):
            axes[2].annotate(f"{row['nll_mean']:.4f}", (x, row['nll_mean']), xytext=(0, 9),
                textcoords='offset points', ha='center', color=color, fontsize=9)
    axes[2].set(xticks=[0, 1], xticklabels=['Raw', 'Temperature\ncalibrated'],
        xlim=(-.25, 1.25), ylim=(0, .23), ylabel='Mean negative log likelihood', title='Mixed models: same calibration pool')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle('Fresh CLINC: 4,197 identical oracle-eight questions; no test fitting', fontsize=12, y=.94)
    fig.text(.5, .035, 'SD describes three observed seeds; paired intervals resample source-question groups conditional on those seeds.\n'
        'Mixed calibration uses 3,396 held-out items. Capacity, pretraining and training exposure differ between families.',
        ha='center', fontsize=8.5)
    output = ROOT / 'paper/figures'
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / 'fresh_clinc_expansion.png', dpi=180)
    fig.savefig(output / 'fresh_clinc_expansion.pdf')
    plt.close(fig)
    (output / 'fresh_clinc_expansion-evidence.json').write_text(json.dumps({
        'source': str(source.relative_to(ROOT)), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'dataset_sha256': report['dataset_sha256'], 'items': report['items'], 'seeds': [42, 43, 44],
        'local_rows': [r for r in report['rows'] if not r['arm'].startswith('community-')],
        'paired_expansion': evidence,
        'scope': 'Observed supervised domain expansion; no architecture causality or universal superiority. SD is not CI; calibration panel compares mixed models only.'
    }, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
