"""Freeze full-label SNIPS/AG News benchmarks before scoring any systems."""
import hashlib
import json
from pathlib import Path
import random
import re
from collections import Counter
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/expansion-v1'
SOURCES = {'snips': ('DeepPavlov/snips', '45f42ebd9641832fd31137317ca5fc5885c86094'),
           'agnews': ('fancyzhx/ag_news', 'eb185aade064a813bc0b7f42de02595523103ca4')}

def normal(text):
    return ' '.join(text.lower().split())

def main():
    if OUT.exists():
        raise FileExistsError('Preserve frozen benchmark; inspect existing manifest')
    OUT.mkdir(parents=True)
    for task, (repo, revision) in SOURCES.items():
        raw = {}
        raw_records = {}
        files = ['data/train-00000-of-00001.parquet', 'data/test-00000-of-00001.parquet']
        if task == 'snips': files.append('intents/intents-00000-of-00001.parquet')
        for name in files:
            path = Path(hf_hub_download(repo, name, repo_type='dataset', revision=revision,
                local_dir=str(ROOT / 'data/expansion-source' / task)))
            raw[name.split('/')[0] if name.startswith('intents/') else name.split('/')[-1].split('-')[0]] = pq.read_table(path).to_pylist()
            raw_records[name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
        if task == 'snips':
            labels = {r['id']: r['name'] for r in raw['intents']}
            descriptions = {str(k): re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', v).lower() for k,v in sorted(labels.items())}
            text_key = 'utterance'
            question = 'Which intent best describes the user request?'
        else:
            labels = {0:'World',1:'Sports',2:'Business',3:'Science and Technology'}
            descriptions = {str(k):v for k,v in labels.items()}
            text_key = 'text'
            question = 'Which news topic best describes this article?'
        records = {}
        for split in ['train','test']:
            records[split] = [{'id': f'{task}-{split}-{i}', 'group_id': hashlib.sha256(normal(r[text_key]).encode()).hexdigest(),
                'split':split,'state':r[text_key], 'question':question,'criteria':descriptions,'label':str(r['label']),
                'oracle_shortlist':False,'full_label_set':True,'domain':task} for i,r in enumerate(raw[split])]
        # Preserve the entire official test split; remove test/dev text from training.
        test_groups = {r['group_id'] for r in records['test']}
        train = [r for r in records['train'] if r['group_id'] not in test_groups]
        seen = set(); dedup=[]; conflicts=0
        labels_by_group={}
        for r in train:
            if r['group_id'] in seen:
                conflicts += labels_by_group[r['group_id']] != r['label']; continue
            seen.add(r['group_id']);labels_by_group[r['group_id']]=r['label'];dedup.append(r)
        train=dedup
        rng=random.Random(20261001); dev=[]; fitted=[]
        for label in descriptions:
            rows=[r for r in train if r['label']==label];rng.shuffle(rows)
            n=min(100,max(1,int(len(rows)*.1)))
            dev.extend({**r,'split':'dev'} for r in rows[:n]);fitted.extend(rows[n:])
        records.update(train=fitted,dev=dev)
        folder=OUT/task;folder.mkdir()
        manifest={'source':repo,'revision':revision,'raw_files':raw_records,'labels':labels,'criteria':descriptions,
            'protocol_seed':20261001,'full_label_set':True,'oracle_shortlist':False,'test_preserved':True,
            'upstream_train_rows':len(raw['train']),'removed_train_test_overlap':len(raw['train'])-len([r for r in raw['train'] if hashlib.sha256(normal(r[text_key]).encode()).hexdigest() not in test_groups]),
            'training_duplicate_label_conflicts':conflicts,'splits':{},'status_when_frozen':'unscored',
            'scope':'Public HF dataset variant, complete test split; disjoint normalized text groups. No test-guided recipe selection.'}
        for split, rows in records.items():
            payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows).encode()
            (folder/(split+'.jsonl')).write_bytes(payload)
            manifest['splits'][split]={'items':len(rows),'groups':len({r['group_id'] for r in rows}),
                'sha256':hashlib.sha256(payload).hexdigest(),'label_counts':dict(Counter(r['label'] for r in rows))}
        (folder/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        print(json.dumps({'task':task,'revision':revision,'splits':manifest['splits']}),flush=True)

if __name__=='__main__':main()
