"""Portable staged pilot. Explicit stages; never silently call a paid API."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ['PYTHONUTF8'] = '1'
os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
CFG = json.loads((ROOT / 'configs/single_16gb.json').read_text(encoding='utf-8'))

def run(*args):
    subprocess.run([sys.executable, *map(str, args)], check=True, cwd=ROOT)

def prepare_pilot(source, out):
    if out.exists():
        raise ValueError('Pilot output exists; preserve it and use a new directory')
    splits = {}
    rng = random.Random(CFG['seed'])
    for name in ['train', 'calibration', 'test']:
        rows = [json.loads(s) for s in (source / f'{name}.jsonl').read_text(encoding='utf-8').splitlines() if s]
        rng.shuffle(rows)
        splits[name] = rows
    # Development is from training only; calibration and test remain independent.
    splits['dev'] = [{**r, 'split': 'dev'} for r in splits['train'][:200]]
    splits['train'] = splits['train'][200:1200]
    splits['calibration'] = splits['calibration'][:200]
    splits['test'] = splits['test'][:200]
    groups = set()
    for name, rows in splits.items():
        if not rows:
            raise ValueError(f'Empty {name} split')
        ids = {r.get('group_id', r['id']) for r in rows}
        if groups & ids:
            raise ValueError('Group leakage between splits')
        groups |= ids
    out.mkdir(parents=True)
    hashes = {}
    for name, rows in splits.items():
        path = out / f'{name}.jsonl'
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / 'manifest.json').write_text(json.dumps({'config': CFG, 'counts': {k: len(v) for k,v in splits.items()},
        'sha256': hashes, 'source_manifest': json.loads((source / 'manifest.json').read_text(encoding='utf-8'))}, indent=2), encoding='utf-8')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['lock', 'prepare', 'encoder-train', 'encoder-eval', 'decoder-base', 'sft-train', 'sft-eval', 'dynamic', 'litjev', 'kev', 'jev'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--split', choices=['dev', 'calibration', 'test'], default='dev')
    p.add_argument('--epochs', type=int, help='Development-only training override; recorded in checkpoint metadata')
    p.add_argument('--tag', default='', help='Suffix for fresh checkpoint/result paths, e.g. e3')
    p.add_argument('--readout', choices=['scalar', 'pointer'], default='scalar')
    p.add_argument('--input-mode', choices=['joint', 'independent'], default='joint')
    p.add_argument('--head-lr', type=float)
    p.add_argument('--data-dir', default='data/pilot16', help='Frozen split directory; use fresh tags for new training data')
    p.add_argument('--jev-model', help='Explicit available official model version, not an assumed latest')
    a = p.parse_args()
    if (a.epochs is not None and a.epochs < 1) or (a.tag and not re.fullmatch(r'[A-Za-z0-9_-]+', a.tag)):
        p.error('epochs must be positive; tag may contain only letters, digits, underscores and hyphens')
    lock_path = ROOT / CFG.get('lock_file', 'configs/resolved_revisions.json')
    if a.stage == 'lock':
        from huggingface_hub import HfApi
        api = HfApi()
        lock = {k: {'id': CFG[k], 'revision': (api.dataset_info(CFG[k]) if k == 'dataset' else api.model_info(CFG[k])).sha}
                for k in ['encoder', 'decoder', 'dataset']}
        with lock_path.open('x', encoding='utf-8') as f:
            json.dump(lock, f, indent=2)
        print(lock_path)
        return
    lock = json.loads(lock_path.read_text(encoding='utf-8'))
    if a.stage == 'prepare':
        source = Path(CFG.get('source_data', 'data/banking77-8'))
        if not source.exists():
            run('scripts/prepare_banking77.py', '--dataset', lock['dataset']['id'], '--revision', lock['dataset']['revision'], '--candidates', CFG['candidates'], '--out', source)
        manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
        if manifest['dataset'] != lock['dataset']['id'] or manifest['revision'] != lock['dataset']['revision'] or manifest['candidates'] != CFG['candidates']:
            raise ValueError('Source data does not match pinned revision/candidate count')
        prepare_pilot(source, Path('data/pilot16'))
        return
    data = str(Path(a.data_dir) / f'{a.split}.jsonl')
    tag = f's{a.seed}'
    if a.tag:
        tag += '-' + a.tag
    encoder = ['--model', lock['encoder']['id'], '--revision', lock['encoder']['revision']]
    decoder = ['--model', lock['decoder']['id'], '--revision', lock['decoder']['revision']]
    train = ['--data', str(Path(a.data_dir) / 'train.jsonl'), '--epochs', str(a.epochs or CFG['epochs']), '--seed', str(a.seed), '--max-length', str(CFG['max_length']), '--device', 'cuda']
    if a.stage == 'encoder-train':
        head = ['--head-lr', a.head_lr] if a.head_lr is not None else []
        run('-m', 'decision_lab', 'train', *encoder, *train, '--readout', a.readout, '--input-mode', a.input_mode, *head, '--batch-size', CFG['encoder_batch_size'], '--out', f'checkpoints/encoder16-{tag}')
    elif a.stage == 'encoder-eval':
        run('-m', 'decision_lab', 'evaluate', '--backend', 'encoder', '--checkpoint', f'checkpoints/encoder16-{tag}', '--data', data, '--device', 'cuda', '--out', f'results/encoder16-{tag}-{a.split}')
    elif a.stage == 'sft-train':
        run('-m', 'prefill_renorm_sft', 'train', *decoder, *train, '--objective', 'candidate_ce', '--rank', CFG['lora_rank'], '--accumulation', CFG['gradient_accumulation'], '--dtype', CFG['dtype'], '--out', f'checkpoints/sft16-{tag}')
    elif a.stage in {'decoder-base', 'sft-eval'}:
        extra = ['--adapter', f'checkpoints/sft16-{tag}'] if a.stage == 'sft-eval' else decoder
        run('-m', 'prefill_renorm_sft', 'evaluate', *extra, '--data', data, '--device', 'cuda', '--seed', a.seed, '--max-length', CFG['max_length'], '--dtype', CFG['dtype'], '--out', f'results/{a.stage}-{tag}-{a.split}')
    elif a.stage == 'dynamic':
        for condition in ['original', 'shuffle', 'missing']:
            run('-m', 'dynamic_candidates.experiment', *decoder, '--data', data, '--device', 'cuda', '--condition', condition, '--seed', a.seed, '--max-length', CFG['max_length'], '--out', f'results/dynamic-{condition}-{tag}-{a.split}')
    elif a.stage == 'litjev':
        run('-m', 'prefill_renorm_sft.litjev_optimization.benchmark', *decoder, '--device', 'cuda', '--dtype', CFG['dtype'], '--questions', 1, 2, '--state-chars', 64, '--repeats', 20, '--out', f'results/litjev16-{tag}.json')
    else:
        if a.stage == 'jev' and not a.jev_model:
            p.error('Supply --jev-model; this stage sends data to a paid external API')
        run('-m', 'decision_lab', 'evaluate', '--backend', a.stage, '--model', a.jev_model if a.stage == 'jev' else 'kev-latest', '--data', data, '--out', f'results/{a.stage}16-{a.split}')

if __name__ == '__main__':
    main()
