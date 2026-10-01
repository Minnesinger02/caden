"""Publish an explicitly authorized encoder update with optimistic revision checks."""
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'release-staging/caden-v2'
MODEL=STAGE/'huggingface-encoder'

def main():
    manifest=json.loads((STAGE/'model-manifest.json').read_text(encoding='utf-8'))
    expected={r['path']:r for r in manifest['files']}
    files={p.relative_to(MODEL).as_posix():p for p in MODEL.rglob('*') if p.is_file()}
    assert set(files)==set(expected)
    for name,p in files.items():assert hashlib.sha256(p.read_bytes()).hexdigest()==expected[name]['sha256']
    validation=json.loads((STAGE/'local-validation.json').read_text(encoding='utf-8'))
    assert validation['load_and_inference_passed'] and validation['manifest_verified']
    api=HfApi();assert api.whoami()['name'].lower()=='leonard02'
    info=api.model_info(manifest['repo']);assert not info.private
    assert info.sha==manifest['previous_revision'],'Remote changed; inspect before applying reviewed update'
    commit=api.upload_folder(repo_id=manifest['repo'],folder_path=str(MODEL),repo_type='model',parent_commit=info.sha,
        commit_message='Release Caden v2: three-seed multi-domain continuation, calibration and audited accuracy/variance/speed')
    info=api.model_info(manifest['repo']);assert info.sha==commit.oid and not info.private
    remote={r.path:r for r in api.list_repo_tree(manifest['repo'],revision=commit.oid,recursive=True) if hasattr(r,'blob_id')}
    assert set(remote)-{'.gitattributes'}==set(expected)
    for name,r in expected.items():
        item=remote[name];assert item.size==r['bytes']
        if item.lfs:assert item.lfs.sha256==r['sha256']
        else:
            data=files[name].read_bytes();assert item.blob_id==hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    previous=api.model_info(manifest['repo'],revision=manifest['previous_revision']);assert previous.sha==manifest['previous_revision']
    record={'repo':manifest['repo'],'url':'https://huggingface.co/'+manifest['repo'],'revision':commit.oid,
        'previous_revision':manifest['previous_revision'],'previous_version_accessible':True,'files_verified':len(expected),'public':True,'verified':True}
    (STAGE/'publication-model-completed.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    manifest.update(published=True,publication=record);(STAGE/'model-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
