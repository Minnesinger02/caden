"""Publication figure of measured context accuracy and repeated-premise cost."""
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
    source = ROOT / 'results/context-ablation-summary.json'
    report = json.loads(source.read_text(encoding='utf-8'))
    rows = report['rows']
    if [r['mode'] for r in rows] != ['joint', 'independent'] or report['seeds'] != [42, 43, 44]:
        raise ValueError('Expected complete matched three-seed context arms')
    output = ROOT / 'paper/figures'
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.6), constrained_layout=True)
    x = np.arange(2)
    labels = ['Joint', 'Independent']
    colors = ['#0072B2', '#D55E00']
    for i, r in enumerate(rows):
        axes[0].errorbar(i, 100*r['accuracy_mean'], yerr=100*r['accuracy_seed_sd'],
                        fmt='o', color=colors[i], markersize=8, capsize=5, linewidth=2)
    axes[0].set(xticks=x, xticklabels=labels, ylim=(85, 100), ylabel='Development accuracy (%)', title='Accuracy: 3 seeds, mean ± SD')
    for i, r in enumerate(rows):
        axes[0].text(i, 86, f"{100*r['accuracy_mean']:.2f}%", ha='center', color=colors[i], fontsize=10)
    axes[0].set_xlim(-.5, 1.5)
    valid = [r['workload_means']['valid_tokens'] for r in rows]
    padding = [r['workload_means']['padded_tokens'] - v for r, v in zip(rows, valid)]
    axes[1].bar(x, valid, color=colors, width=.55, label='Nonpadding')
    axes[1].bar(x, padding, bottom=valid, color='#BBBBBB', width=.55, label='Padding')
    axes[1].set(xticks=x, xticklabels=labels, ylim=(0, 470), ylabel='Encoder tokens / question', title='Actual input at B=1 question')
    for i, r in enumerate(rows):
        w = r['workload_means']
        axes[1].text(i, w['padded_tokens'] + 12, f"{w['padded_tokens']:.1f}; {w['encoder_rows']:.0f} row{'s' if w['encoder_rows'] > 1 else ''}", ha='center', fontsize=9)
    axes[1].legend(frameon=False, fontsize=9, loc='upper left')
    axes[2].bar(x, [r['latency_p50_mean_ms'] for r in rows], color=colors, width=.55)
    axes[2].set(xticks=x, xticklabels=labels, ylim=(0, 5.6), ylabel='Mean run p50 (ms)', title='Historical serial timing')
    for i, r in enumerate(rows):
        axes[2].text(i, r['latency_p50_mean_ms'] + .15, f"{r['latency_p50_mean_ms']:.2f} ms", ha='center')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle('Candidate context: same 5,000 training items and 200 oracle-eight development questions', fontsize=11)
    fig.savefig(output / 'context_accuracy_cost.png', dpi=180)
    fig.savefig(output / 'context_accuracy_cost.pdf')
    plt.close(fig)
    (output / 'context_accuracy_cost-evidence.json').write_text(json.dumps({'source': str(source.relative_to(ROOT)),
        'dataset_sha256': report['dataset_sha256'], 'train_sha256': report['train_sha256'], 'seeds': report['seeds'],
        'scope': 'exploratory development accuracy; seed SD not confidence interval; real tokenizer accounting includes repeated premise and padding; historical p50 not replicated timing'}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
