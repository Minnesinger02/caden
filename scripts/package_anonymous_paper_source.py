"""Prepare a minimal paper source bundle; do not certify full submission anonymity."""
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    paper = ROOT / 'paper/arr'
    files = {name: (paper / name).read_bytes() for name in
             ['main_en.tex', '中文对应稿.md', 'acl.sty', 'acl_natbib.bst']}
    digest = hashlib.sha256(files['main_en.tex']).hexdigest()
    translation = json.loads((paper / 'translation-audit.json').read_text(encoding='utf-8'))
    if translation['english_master_sha256'] != digest or translation['chinese_sha256'] != hashlib.sha256(files['中文对应稿.md']).hexdigest():
        raise ValueError('Translation snapshot mismatch')
    template = json.loads((paper / 'template-lock.json').read_text(encoding='utf-8'))
    for name in ['acl.sty', 'acl_natbib.bst']:
        if hashlib.sha256(files[name]).hexdigest() != template['files'][name]:
            raise ValueError('Official style changed')
    files['TEMPLATE_PROVENANCE.json'] = json.dumps(template, indent=2).encode()
    files['README.md'] = (
        '# Anonymous paper source preparation\n\n'
        'English master: main_en.tex; Chinese counterpart: 中文对应稿.md. Official ACL review style files are unchanged. '
        'Bibliography and all four tables are embedded; no external figure or bibliography file is needed.\n\n'
        'No verified manuscript PDF is supplied. Native compilation failed: Unable to find standard directories for platform. '
        'Main-text page count, floats, hyperlinks, fonts and full review anonymity require rendered inspection. '
        'The Chinese text is a reading counterpart, not the ARR submission PDF.\n\n'
        'This source-only preparation excludes private logs, absolute local paths, raw data, weights and author accounts. '
        'The literal scan checks specified patterns and author settings; it does not certify complete anonymity, rights or human author review. '
        'Numerical reproducibility requires a separately reviewed anonymous supplement. This is not a completed submission.\n'
    ).encode('utf-8')
    patterns = [r'(?i)(?<![A-Za-z0-9])[A-Z]:[\\/]', r'(?i)C[\\/]Users', r'(?i)chatgpt\.com/s/',
                r'(?i)(?:sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|apikey_[A-Za-z0-9_]{20,})']
    for name, data in files.items():
        if any(re.search(pattern, data.decode('utf-8-sig')) for pattern in patterns):
            raise ValueError(f'Explicit local path/token/share-link pattern in {name}')
    source = files['main_en.tex'].decode('utf-8-sig')
    if re.findall(r'\\author\{([^}]+)\}', source) != ['Anonymous ACL submission']:
        raise ValueError('Anonymous author declaration changed')
    if re.search(r'\\(?:input|include|includegraphics|bibliography)\s*(?:\[[^\]]*\])?\s*\{', source):
        raise ValueError('Unpackaged manuscript input')
    report = {'english_master_sha256': digest, 'files': [
        {'path': name, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)} for name, data in sorted(files.items())],
        'literal_scan_passed': True, 'scan_scope': 'Drive-letter/local user paths, chat share links, common API-token patterns, anonymous author and external inputs.',
        'full_anonymity_certified': False, 'pdf_verified': False, 'submission_ready': False}
    out = ROOT / 'paper/submission-source'
    out.mkdir(exist_ok=True)
    package = out / ('paper-source-' + digest[:16] + '.zip')
    with zipfile.ZipFile(package, 'x', zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            archive.writestr(name, data)
        archive.writestr('SOURCE_MANIFEST.json', json.dumps(report, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(package) as archive:
        if archive.testzip():
            raise ValueError('Source CRC failed')
        for item in report['files']:
            if hashlib.sha256(archive.read(item['path'])).hexdigest() != item['sha256']:
                raise ValueError('Source hash mismatch')
    report.update(archive=package.relative_to(ROOT).as_posix(), archive_sha256=hashlib.sha256(package.read_bytes()).hexdigest(),
                  archive_bytes=package.stat().st_size, all_file_hashes_verified=True)
    package.with_suffix('.audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
