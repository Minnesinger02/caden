"""Download the official GitHub CLI into the workspace, with official SHA check."""
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    headers = {'User-Agent': 'Caden-workspace-setup'}
    def fetch(url):
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=45) as response:
            return response.read()
    release = json.loads(fetch('https://api.github.com/repos/cli/cli/releases/latest'))
    asset = next(a for a in release['assets'] if a['name'].endswith('windows_amd64.zip'))
    checks = next(a for a in release['assets'] if a['name'].endswith('checksums.txt'))
    checksum_text = fetch(checks['browser_download_url']).decode()
    expected = next(line.split()[0] for line in checksum_text.splitlines() if line.split()[-1].lstrip('*') == asset['name'])
    data = fetch(asset['browser_download_url'])
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ValueError('Official GitHub CLI checksum mismatch')
    folder = ROOT / '.tools/github-cli' / release['tag_name']
    folder.mkdir(parents=True, exist_ok=False)
    archive = folder / asset['name']
    archive.write_bytes(data)
    with zipfile.ZipFile(archive) as source:
        for item in source.infolist():
            if not (folder / item.filename).resolve().is_relative_to(folder.resolve()):
                raise ValueError('Archive path escapes named install directory')
        source.extractall(folder)
    executable = next(folder.rglob('gh.exe'))
    report = {'version': release['tag_name'], 'sha256': actual, 'executable': str(executable), 'source': asset['browser_download_url']}
    (ROOT / '.tools/github-cli/installed.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
