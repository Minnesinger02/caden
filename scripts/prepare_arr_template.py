"""Download and pin official ACL/ARR style files without a TeX installation."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
out = ROOT/'paper/arr'
out.mkdir(exist_ok=True)
def get(url):
    request = urllib.request.Request(url, headers={'User-Agent':'jev-arr-research'})
    with urllib.request.urlopen(request, timeout=60) as stream:
        return stream.read()
repo = 'acl-org/acl-style-files'
metadata = json.loads(get(f'https://api.github.com/repos/{repo}'))
sha = json.loads(get(f'https://api.github.com/repos/{repo}/commits/{metadata["default_branch"]}'))['sha']
files = json.loads(get(f'https://api.github.com/repos/{repo}/contents?ref={sha}'))
report = {'repo':repo, 'revision':sha, 'author_guidelines':'https://aclrollingreview.org/cfp', 'files':{}}
for name in ['acl.sty','acl_natbib.bst','acl_latex.tex','README.md']:
    if not any(f['name']==name for f in files):
        if name in ['acl.sty','acl_natbib.bst']:
            raise FileNotFoundError(name)
        continue
    data = get(f'https://raw.githubusercontent.com/{repo}/{sha}/{name}')
    target = out/name
    if target.exists() and target.read_bytes()!=data:
        raise FileExistsError(target)
    target.write_bytes(data)
    report['files'][name] = hashlib.sha256(data).hexdigest()
(out/'template-lock.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
