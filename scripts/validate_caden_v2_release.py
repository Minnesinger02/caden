"""Check staged bytes and real CPU loading for all updated encoder seeds."""
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from decision_lab.encoder import CandidateEncoder
from decision_lab.common import validate_probs
STAGE=ROOT/'release-staging/caden-v2'

def main():
    torch.set_num_threads(2)
    manifest=json.loads((STAGE/'model-manifest.json').read_text())
    for entry in manifest['files']:
        path=STAGE/'huggingface-encoder'/entry['path']
        assert path.stat().st_size==entry['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
    rows=[{'state':'A customer wants to replace a missing bank card.','question':'Which intent best describes this request?',
        'criteria':{'replace':'replace a lost bank card','music':'play a music track','sports':'sports news'}},
        {'state':'The football team won the league championship.','question':'Which topic best describes the article?',
        'criteria':{'0':'world news','1':'sports','2':'business','3':'science and technology'}}]
    results=[]
    for seed in [42,43,44]:
        folder=STAGE/f'huggingface-encoder/seed-{seed}';model=CandidateEncoder.load(folder).eval()
        cal=json.loads((folder/'calibration.json').read_text());assert cal['valid_items']==3996 and cal['temperature']>0
        for r in rows:
            with torch.inference_mode():z=model([r])[0];p=(z/cal['temperature']).softmax(-1).tolist()
            assert torch.isfinite(z).all();validate_probs(dict(zip(r['criteria'],p)),r['criteria'])
        results.append({'seed':seed,'inference_rows':len(rows),'cpu_load_passed':True,'temperature':cal['temperature']})
        del model
    report={'manifest_verified':True,'load_and_inference_passed':True,'seeds':results,'scope':'Actual CPU model loading/probability support checks, not a benchmark or deployment calibration guarantee.'}
    (STAGE/'local-validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
