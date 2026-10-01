"""Verified small-range downloads for public checkpoints on unstable connections."""
import hashlib
import argparse
import json
from pathlib import Path
import time
import requests
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--lock', default='configs/community-models-lock.json')
parser.add_argument('--report', default='handoff/community-checkpoints.json')
args = parser.parse_args()
locks = json.loads((ROOT/args.lock).read_text(encoding='utf-8'))
report_path = ROOT/args.report
report = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {}
session = requests.Session()
for name, info in locks.items():
    destination = ROOT/'.cache/community'/name.replace('/','--')/info['revision']
    destination.mkdir(parents=True, exist_ok=True)
    model_info = HfApi().model_info(name, revision=info['revision'], files_metadata=True)
    files = {f.rfilename: f for f in model_info.siblings}
    chosen = [f for f in info['files'] if f in ['README.md','LICENSE','option_marker.pt'] or
              (f.endswith(('.json','.safetensors','.py')) and not f.startswith(('assets/','eval/','multilingual/','typed-decisions/')))]
    for filename in chosen:
        metadata = files[filename]
        if not filename.endswith(('.safetensors','.pt')):
            hf_hub_download(name, filename, revision=info['revision'], local_dir=destination)
            continue
        size = metadata.size
        expected = metadata.lfs.sha256 if metadata.lfs else None
        target = destination/filename
        if target.exists() and target.stat().st_size == size:
            with target.open('rb') as stream:
                if expected and hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                    raise ValueError(f'Existing file hash mismatch: {name}/{filename}')
            continue
        partial = target.with_suffix(target.suffix+'.range-part')
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > size:
            raise ValueError('Partial file exceeds expected size')
        print(f'{name}/{filename}: resuming {offset}/{size} bytes', flush=True)
        while offset < size:
            end = min(size-1, offset+2*1024*1024-1)
            url = f'https://huggingface.co/{name}/resolve/{info["revision"]}/{filename}'
            for attempt in range(6):
                try:
                    response = session.get(url, params={'download':'true', 'range_start':str(offset)},
                                           headers={'Range':f'bytes={offset}-{end}'}, timeout=(30,90), stream=True)
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {offset}-{end}/{size}':
                        response.close()
                        raise ValueError('Server did not honor exact requested range; refusing corrupt append')
                    content = response.content
                    response.close()
                    if len(content) != end-offset+1:
                        raise ValueError('Short range response')
                    with partial.open('ab') as stream:
                        stream.write(content)
                    offset = end+1
                    if offset % (16*1024*1024)==0 or offset==size:
                        print(f'{name}/{filename}: {offset}/{size} bytes', flush=True)
                    break
                except (requests.RequestException, ValueError) as error:
                    if attempt==5:
                        raise RuntimeError(f'Range download failed at {offset}: {type(error).__name__}') from error
                    print(f'Retry range {offset}, attempt {attempt+1}: {type(error).__name__}', flush=True)
                    time.sleep(min(attempt+1, 5))
        with partial.open('rb') as stream:
            actual = hashlib.file_digest(stream,'sha256').hexdigest()
        if expected and actual != expected:
            raise ValueError('Complete checkpoint SHA256 mismatch')
        partial.replace(target)
    report[name] = {'revision':info['revision'], 'path':str(destination), 'files':chosen}
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Verified {name}', flush=True)
