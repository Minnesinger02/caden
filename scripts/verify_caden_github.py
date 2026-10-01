"""Verify the authorized public release against the locally committed Git tree."""
import json
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
stage = root / 'release-staging/caden'
code = stage / 'github'
gh = json.loads((root / '.tools/github-cli/installed.json').read_text())['executable']
os.environ.update(GIT_CONFIG_COUNT='1', GIT_CONFIG_KEY_0='safe.directory', GIT_CONFIG_VALUE_0=code.as_posix())

def run(*args):
    return subprocess.check_output(args, cwd=code, text=True, encoding='utf-8')

assert json.loads(run(gh, 'api', 'user'))['login'].lower() == 'minnesinger02'
settings = json.loads(run(gh, 'api', 'repos/Minnesinger02/caden'))
assert settings['has_issues'] is False and settings['private'] is False
head = run('git', 'rev-parse', 'HEAD').strip()
ref = json.loads(run(gh, 'api', 'repos/Minnesinger02/caden/git/ref/heads/main'))
assert ref['object']['sha'] == head
tree = json.loads(run(gh, 'api', f'repos/Minnesinger02/caden/git/trees/{head}?recursive=1'))
assert not tree['truncated']
remote = {e['path']: e['sha'] for e in tree['tree'] if e['type'] == 'blob'}
local = {}
for line in run('git', 'ls-tree', '-r', 'HEAD').splitlines():
    metadata, path = line.split('\t', 1)
    local[path] = metadata.split()[2]
manifest = json.loads((stage / 'release-manifest.json').read_text(encoding='utf-8'))
assert set(local) == {e['path'] for e in manifest['packages']['github']['files']}
assert local == remote
progress = {'github_url': settings['html_url'], 'git_commit': head, 'models': [],
            'github_verified': True, 'github_has_issues': False, 'github_files_verified': len(remote)}
for name in ['publication-progress.json', 'publication-github-completed.json']:
    (stage / name).write_text(json.dumps(progress, indent=2), encoding='utf-8')
print(json.dumps(progress, indent=2))
