"""Publish the reviewed Caden allowlist, after explicit local CLI authentication."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'release-staging/caden'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--publish', action='store_true', help='Create public repos and upload the reviewed release')
    parser.add_argument('--target', choices=['all', 'github', 'huggingface'], default='all')
    args = parser.parse_args()
    manifest = json.loads((STAGE / 'release-manifest.json').read_text(encoding='utf-8'))
    for folder, package in manifest['packages'].items():
        current = {p.relative_to(STAGE / folder).as_posix() for p in (STAGE / folder).rglob('*')
                   if p.is_file() and not any(part in {'.git', '__pycache__'} for part in p.relative_to(STAGE / folder).parts) and p.suffix != '.pyc'}
        if current != {entry['path'] for entry in package['files']}:
            raise ValueError('Release file list changed; review before publishing')
        for entry in package['files']:
            if sha(STAGE / folder / entry['path']) != entry['sha256']:
                raise ValueError('Release content changed; review before publishing')
    if not args.publish:
        print(json.dumps({'preflight_passed': True, 'github': manifest['github_repo'], 'has_issues': False,
            'encoder': manifest['encoder_repo'], 'qwen': manifest['qwen_repo'], 'visibility': 'public', 'uploaded': False}, indent=2))
        return
    install = json.loads((ROOT / '.tools/github-cli/installed.json').read_text(encoding='utf-8'))
    gh = install['executable']
    # The sandbox-created checkout has a different Windows SID. Trust only this
    # known release directory for this process; do not change global Git trust.
    config_index = int(os.environ.get('GIT_CONFIG_COUNT', '0'))
    os.environ['GIT_CONFIG_COUNT'] = str(config_index + 1)
    os.environ[f'GIT_CONFIG_KEY_{config_index}'] = 'safe.directory'
    os.environ[f'GIT_CONFIG_VALUE_{config_index}'] = (STAGE / 'github').as_posix()
    def run(command, cwd=None):
        return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True).stdout
    user = None
    if args.target != 'huggingface':
        user = json.loads(run([gh, 'api', 'user']))
        if user['login'].lower() != 'minnesinger02':
            raise ValueError('GitHub authenticated owner differs from authorized user')
    api = None
    if args.target != 'github':
        from huggingface_hub import HfApi
        api = HfApi()
        if api.whoami()['name'].lower() != 'leonard02':
            raise ValueError('Hugging Face authenticated owner differs from authorized user')
    # Check both authenticated identities before performing any external mutation.
    progress_path = STAGE / 'publication-progress.json'
    published = json.loads(progress_path.read_text(encoding='utf-8')) if progress_path.exists() else {'models': []}
    code = STAGE / 'github'
    if args.target == 'huggingface':
        publish_models(api, manifest, published)
        return
    if not (code / '.git').exists():
        run(['git', 'init', '-b', 'main'], cwd=code)
    head = subprocess.run(['git', 'rev-parse', '--verify', 'HEAD'], cwd=code, capture_output=True, text=True)
    if head.returncode != 0:
        run(['git', 'config', 'user.name', user['login']], cwd=code)
        run(['git', 'config', 'user.email', f"{user['id']}+{user['login']}@users.noreply.github.com"], cwd=code)
        run(['git', 'add', '.'], cwd=code)
        run(['git', 'commit', '-m', 'Release Caden source and documented model loaders'], cwd=code)
    # Creation sets Issues off before the first push. Existing remotes are not force-pushed.
    remote_check = subprocess.run([gh, 'repo', 'view', manifest['github_repo'], '--json', 'name'], capture_output=True, text=True)
    if remote_check.returncode == 0:
        raise ValueError('Target GitHub repository already exists; inspect and resume explicitly without force push')
    run([gh, 'repo', 'create', manifest['github_repo'], '--public', '--disable-issues', '--source', str(code), '--remote', 'origin'])
    run([gh, 'repo', 'edit', manifest['github_repo'], '--enable-issues=false'])
    settings = json.loads(run([gh, 'api', 'repos/' + manifest['github_repo']]))
    if settings['has_issues'] or settings['private']:
        raise ValueError('GitHub requested publication settings were not applied')
    run([gh, 'auth', 'setup-git'])
    run(['git', 'push', '-u', 'origin', 'main'], cwd=code)
    published = {'github_url': settings['html_url'], 'git_commit': run(['git', 'rev-parse', 'HEAD'], cwd=code).strip(), 'models': []}
    (STAGE / 'publication-progress.json').write_text(json.dumps(published, indent=2), encoding='utf-8')
    if args.target == 'github':
        final = json.loads(run([gh, 'api', 'repos/' + manifest['github_repo']]))
        if final['has_issues']:
            raise ValueError('Final GitHub Issues verification failed')
        published.update(github_has_issues=False, github_verified=True)
        (STAGE / 'publication-github-completed.json').write_text(json.dumps(published, indent=2), encoding='utf-8')
        progress_path.write_text(json.dumps(published, indent=2), encoding='utf-8')
        print(json.dumps(published, indent=2))
        return
    publish_models(api, manifest, published)
    final = json.loads(run([gh, 'api', 'repos/' + manifest['github_repo']]))
    if final['has_issues']:
        raise ValueError('Final GitHub Issues verification failed')
    published.update(github_has_issues=False, verified=True)
    (STAGE / 'publication-completed.json').write_text(json.dumps(published, indent=2), encoding='utf-8')
    print(json.dumps(published, indent=2))


def publish_models(api, manifest, published):
    for repo, folder in [(manifest['encoder_repo'], 'huggingface-encoder'), (manifest['qwen_repo'], 'huggingface-qwen-lora')]:
        api.create_repo(repo, repo_type='model', private=False, exist_ok=False)
        commit = api.upload_folder(repo_id=repo, repo_type='model', folder_path=str(STAGE / folder),
            commit_message='Release three recorded training seeds, calibration, provenance and model card')
        info = api.model_info(repo)
        if info.private or info.sha != commit.oid:
            raise ValueError('Hugging Face published revision/visibility mismatch')
        published['models'].append({'repo': repo, 'commit': commit.oid, 'url': 'https://huggingface.co/' + repo})
        (STAGE / 'publication-progress.json').write_text(json.dumps(published, indent=2), encoding='utf-8')
    published['huggingface_verified'] = True
    if published.get('github_verified') and published.get('github_has_issues') is False:
        published['verified'] = True
        (STAGE / 'publication-completed.json').write_text(json.dumps(published, indent=2), encoding='utf-8')
    (STAGE / 'publication-progress.json').write_text(json.dumps(published, indent=2), encoding='utf-8')
    print(json.dumps(published, indent=2))


if __name__ == '__main__':
    main()
