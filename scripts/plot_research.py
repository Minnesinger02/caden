"""Scientific figures generated from saved metrics with seed counts explicit."""
import json
from pathlib import Path
import statistics
import os
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1]/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / 'paper/figures'
out.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'figure.dpi': 140})
def measured(template):
    return [json.loads(p.read_text(encoding='utf-8')) for seed in [42,43,44]
            if (p := ROOT / 'results' / template.format(seed=seed) / 'metrics.json').exists()]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), constrained_layout=True)
evidence = {}
scales = [1000, 5000, 8792]
for label, color, templates in [('DistilBERT scalar', '#0072B2', ['encoder16-s{seed}-e3-dev', 'encoder16-s{seed}-e3-5k-dev', 'encoder16-s{seed}-e3-8792-dev']),
                                 ('Qwen3 LoRA', '#D55E00', ['sft-eval-s{seed}-dev', 'sft-eval-s{seed}-train5k-dev', 'sft-eval-s{seed}-train8792-dev'])]:
    acc = [[m['accuracy_all_failures_wrong']*100 for m in measured(template)] for template in templates]
    evidence[label] = {'accuracy': acc, 'seeds_per_point': [len(a) for a in acc]}
    means = [statistics.mean(a) for a in acc]
    errors = [statistics.stdev(a) if len(a)>1 else 0 for a in acc]
    axes[0].errorbar(scales, means, yerr=errors, color=color, marker='o', capsize=4, label=label)
    for x,y,values in zip(scales, means, acc):
        axes[0].annotate(f'n={len(values)}', (x,y), xytext=(7, 9 if label.startswith('Distil') else -17), textcoords='offset points', color=color)
axes[0].set(xlabel='Training items', ylabel='Development accuracy (%)', xticks=scales, ylim=[75,100], title='Oracle 8-option BANKING77; n = training seeds')
axes[0].legend(frameon=False, loc='lower right')
axes[0].grid(axis='y', alpha=.2)
for name,color,file in [('DistilBERT', '#0072B2', 'fixed-compute-encoder-fp32.json'), ('Qwen3 base', '#D55E00', 'fixed-compute-qwen-fp32.json')]:
    rows = json.loads((ROOT/'results'/file).read_text(encoding='utf-8'))['rows']
    rows = [r for r in rows if r['task']=='8-candidate probability readout' and r['input_tokens_per_item']==128]
    axes[1].plot([r['batch'] for r in rows], [r['decisions_per_second'] for r in rows], color=color, marker='o', label=name)
axes[1].set(xlabel='Batch size', ylabel='Decisions / second', xticks=[1,8,16], yscale='log', title='Fixed 128-token GPU-resident FP32 readout')
axes[1].legend(frameon=False)
axes[1].grid(axis='y', which='both', alpha=.2)
for extension in ['png','pdf']:
    fig.savefig(out/f'accuracy_and_throughput.{extension}')
plt.close(fig)
(out/'accuracy_and_throughput-evidence.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
fig,ax = plt.subplots(figsize=(7,4), constrained_layout=True)
for label,color,template in [('Scalar','#0072B2','encoder16-s{seed}-e3-dev'), ('Pointer, higher head LR','#CC79A7','encoder16-s{seed}-pointer-e3-hlr-dev'), ('Qwen3 LoRA','#D55E00','sft-eval-s{seed}-dev')]:
    ms = measured(template)
    xs = [m['accuracy_all_failures_wrong']*100 for m in ms]
    ys = [m['nll_clipped_1e-15'] for m in ms]
    ax.scatter(xs,ys,color=color,s=45,label=label)
    ax.scatter([statistics.mean(xs)],[statistics.mean(ys)],color=color,marker='X',s=95,edgecolors='black',linewidths=.5)
ax.set(xlabel='Development accuracy (%)', ylabel='Negative log likelihood (lower is better)', title='Readout ablation; X = three-seed mean')
ax.legend(frameon=False)
ax.grid(alpha=.2)
for extension in ['png','pdf']:
    fig.savefig(out/f'readout_ablation.{extension}')
plt.close(fig)
print(out)
