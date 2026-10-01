"""Create a private local replication snapshot with hashes; do not publish it."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    paper = ROOT / 'paper/arr'
    master = paper / 'main_en.tex'
    digest = hashlib.sha256(master.read_bytes()).hexdigest()
    translation = json.loads((paper / 'translation-status.json').read_text(encoding='utf-8'))
    if digest != translation['english_master_sha256']:
        raise ValueError('Chinese snapshot differs from English master')
    if not (ROOT / 'handoff/remaining-registered-experiments/completed.json').is_file():
        raise FileNotFoundError('All registered GPU stages must terminate successfully first')
    source_files = {}
    def add(path, archive_name=None):
        if not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError('Archive source resolves outside the project')
        if not path.is_file():
            raise FileNotFoundError(path)
        name = archive_name or path.relative_to(ROOT).as_posix()
        if name in source_files and source_files[name] != path:
            raise ValueError('Archive name collision')
        source_files[name] = path
    def tree(folder, suffixes):
        for path in sorted((ROOT / folder).rglob('*')):
            if not path.is_file() or any(part in {'.git', '__pycache__', '.venv', 'node_modules'} for part in path.relative_to(ROOT / folder).parts):
                continue
            if path.suffix.lower() in suffixes or path.name.upper().startswith(('LICENSE', 'NOTICE')):
                add(path)
    for name in ['pyproject.toml', 'uv.lock', 'README.md', 'START_HERE_WINDOWS.md', 'LICENSE']:
        add(ROOT / name)
    for folder in ['decision_lab', 'dynamic_candidates', 'prefill_renorm_sft', 'scripts', 'tests', 'external']:
        tree(folder, {'.py', '.toml', '.md', '.json', '.yaml', '.yml', '.txt', '.ps1', '.bat', '.jinja'})
    for folder in ['configs', 'data', 'results']:
        tree(folder, {'.json', '.jsonl', '.md', '.csv', '.log', '.txt'})
    for name in ['main_en.tex', '中文对应稿.md', 'acl.sty', 'acl_natbib.bst', 'template-lock.json',
                 'BUILD_STATUS.json', 'translation-source-lock.json', 'translation-status.json', 'translation-audit.json',
                 'manuscript-evidence.json', 'RESPONSIBLE_NLP_DRAFT.md', 'native-compile-diagnostic.json']:
        add(paper / name)
    add(ROOT / 'paper/READING_GUIDE.md')
    add(ROOT / 'handoff/COLLABORATOR_START_HERE.md', 'COLLABORATOR_START_HERE.md')
    tree('handoff', {'.md', '.json', '.log', '.csv', '.txt', '.py'})
    tree('paper/arr/archive', {'.tex', '.md', '.json'})
    tree('paper/figures', {'.png', '.pdf', '.json'})
    tree('paper/submission-source', {'.zip', '.json'})
    for name in ['RESEARCH_PROTOCOL.md', 'FRESH_CLINC_CONFIRMATION_PROTOCOL.md', 'REPLICATED_TIMING_PROTOCOL.md',
                 'CANDIDATE_PERTURBATION_EXECUTION.md', 'RELEASE_PROVENANCE.md', 'TRAINING_EXPOSURE.md',
                 'release-provenance.json', 'training-exposure-declarations.json', 'community-source-snapshots.json',
                 'kev-full-confirmation-overlap.json', 'encoder44-training-timing-gap.json', 'REPRODUCE_WINDOWS.md',
                 'BENCHMARKS.md', 'ASTRA_HANDOFF.md', 'FINAL_RESEARCH_AUDIT.json',
                 'tests-final-registered-experiments.log', 'remaining-registered-queue.log', 'CITATION_REVIEW.md',
                 'RECORDED_TRAINING_BUDGET.json', 'RECORDED_TRAINING_BUDGET.md', 'JEV_API_COMPARISON_PROTOCOL.md',
                 'tests-jev-budget-safety.log']:
        add(ROOT / 'handoff' / name)
    for path in sorted((ROOT / 'handoff').glob('*freeze*.txt')):
        add(path)
    for path in sorted((ROOT / 'handoff').glob('*requirements*resolved*.txt')):
        add(path)
    for path in sorted((ROOT / 'handoff').glob('*checkpoints*.json')):
        add(path)
    tree('handoff/licensing-source', {'.md', '.json', '.py'})
    for folder in ['handoff/remaining-registered-experiments']:
        tree(folder, {'.json'})
    for path in sorted((ROOT / 'checkpoints').rglob('*')):
        if path.is_file() and path.name in ['training.json', 'experiment.json', 'adapter_config.json']:
            add(path, 'checkpoint-recipes/' + path.relative_to(ROOT / 'checkpoints').as_posix())
    manifest = {'english_master_sha256': digest, 'chinese_sha256': hashlib.sha256((paper / '中文对应稿.md').read_bytes()).hexdigest(),
        'scope': 'Private local research snapshot: original source, fixed external source/attribution, frozen datasets, all saved result traces and final paper. No public upload or licensing grant; models/venvs/caches/secrets/chat history excluded.',
        'weights': 'Pretrained and locally trained tensor files excluded. Exact pretrained revisions and original local training recipes retained; train in a fresh workspace to reproduce local checkpoints.',
        'rendering': 'Paper PDF and ARR page limit are unverified due to native compiler platform initialization failure. Figure PDFs are separate verified image artifacts.',
        'jev': 'Pinned jev-1.13.0, 7796 final questions and 7800 paid requests; estimated documented-price expense and full audit in results/jev113-comparison-summary.json. Server hardware/precision/training exposure unknown.', 'files': []}
    for name, path in sorted(source_files.items()):
        manifest['files'].append({'path': name, 'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    source_fingerprint = hashlib.sha256(json.dumps(manifest['files'], sort_keys=True).encode()).hexdigest()
    package = ROOT / 'paper/replication' / ('research-snapshot-' + digest[:12] + '-' + source_fingerprint[:12] + '.zip')
    package.parent.mkdir(parents=True, exist_ok=True)
    if package.exists():
        raise FileExistsError(f'Preserve existing {package}')
    manifest_path = package.with_suffix('.manifest.json')
    with manifest_path.open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
    with zipfile.ZipFile(package, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sorted(source_files.items()):
            archive.write(path, name)
        archive.writestr('REPLICATION_MANIFEST.json', manifest_path.read_bytes())
    with zipfile.ZipFile(package) as archive:
        if archive.testzip() is not None:
            raise ValueError('Archive CRC validation failed')
        for item in manifest['files']:
            if hashlib.sha256(archive.read(item['path'])).hexdigest() != item['sha256']:
                raise ValueError('Archived file hash mismatch')
    report = {'archive': str(package.relative_to(ROOT)), 'bytes': package.stat().st_size,
        'sha256': hashlib.sha256(package.read_bytes()).hexdigest(), 'files': len(manifest['files']),
        'original_bytes': sum(item['bytes'] for item in manifest['files']), 'all_archived_hashes_verified': True,
        'anonymous_manuscript_author': r'\author{Anonymous ACL submission}' in master.read_text(encoding='utf-8'),
        'scope': 'Private local artifact integrity; not public-release legal clearance or guaranteed full-package ARR anonymity/page validation.'}
    package.with_suffix('.audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
