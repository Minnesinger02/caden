"""Preserve revision-pinned card declarations, independently of experiment scores."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / 'handoff/licensing-source'


def main():
    entries = []
    cards = [
        ('BANKING77', 'dataset', 'legacy-datasets/banking77', 'f54121560de48f2852f90be299010d1d6dc612ec', 'cc-by-4.0'),
        ('CLINC150', 'dataset', 'clinc/clinc_oos', '155b9c710419136e17307b80d0a13e68cd46b4ec', 'cc-by-3.0'),
        ('DistilBERT', 'download', 'distilbert/distilbert-base-uncased', '12040accade4e8a0f71eabdb258fecc2e7e948be', 'apache-2.0'),
        ('Qwen3-0.6B', 'download', 'Qwen/Qwen3-0.6B', 'c1899de289a04d12100db370d81485cdf75e47ca', 'apache-2.0'),
        ('Kev', 'download', 'jaredpalmer/kev-0.8b', '9a45d25eb2ab761841196625383fa1dff0e56c1e', 'apache-2.0'),
        ('Laya', 'community', 'convaiinnovations/laya', '55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851', 'apache-2.0'),
        ('Nano', 'community', 'manjunathshiva/opendecider-nano', 'ed8b7d828d97f790db7fb79a09371ef22286a1ae', 'apache-2.0'),
        ('Von', 'community', 'wfzyx/von', '411c44401cccddd792f341edfe033ea834557d13', 'apache-2.0'),
        ('Decider0.8B', 'community', 'Mapika/decider-0.8b', 'a0a01d6f8135298f400a8c856b355793012ae971', 'apache-2.0'),
        ('Decider2B', 'community', 'Mapika/decider-2b', '533964dae8be954c5b5e19fa4948e48408094c1e', 'apache-2.0'),
    ]
    downloaded_names = {'DistilBERT': 'distilbert', 'Qwen3-0.6B': 'qwen06', 'Kev': 'kev08'}
    for name, kind, repo, revision, expected in cards:
        if kind == 'dataset':
            source = ROOT / '.cache/huggingface/hub' / ('datasets--' + repo.replace('/', '--')) / 'snapshots' / revision / 'README.md'
        elif kind == 'community':
            source = ROOT / '.cache/community' / repo.replace('/', '--') / revision / 'README.md'
        else:
            source = DESTINATION / (downloaded_names[name] + '-README.md')
        payload = source.read_bytes()
        text = payload.decode('utf-8').replace('\r\n', '\n')
        front = re.match(r'\A---\n(.*?)\n---', text, re.S)
        if not front:
            raise ValueError(f'No YAML card metadata for {name}')
        license_match = re.search(r'^license:\s*\n?-?\s*([^\n]+)', front.group(1), re.M)
        if license_match is None or license_match.group(1).strip() != expected:
            raise ValueError(f'Unexpected license declaration for {name}')
        target = source if kind == 'download' else DESTINATION / (name + '-README.md')
        if target.exists() and target.read_bytes() != payload:
            raise ValueError(f'Previously saved card differs: {target}')
        target.write_bytes(payload)
        url = f'https://huggingface.co/{"datasets/" if kind == "dataset" else ""}{repo}/blob/{revision}/README.md'
        entries.append({'name': name, 'repo': repo, 'revision': revision, 'card_declared_license': expected,
                        'source_url': url, 'evidence': str(target.relative_to(ROOT)),
                        'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)})
    for name, repo, revision, expected in [
        ('OpenJev source', 'lookski/openjev', 'b52a53963ec7aa6aad571b176a5bef80e3f4150b', 'mit'),
        ('Kev source', 'jaredpalmer/kev', '0fe8fc97c2bcc247fa3efb6e5c32af4e99770e91', 'apache-2.0'),
    ]:
        directory = 'openjev-upstream' if name.startswith('Open') else 'kev-upstream'
        payload = (ROOT / 'external' / directory / 'LICENSE').read_bytes()
        expected_heading = b'MIT License' if expected == 'mit' else b'Apache License'
        if expected_heading not in payload[:160]:
            raise ValueError('Source license heading changed')
        target = DESTINATION / (directory + '-LICENSE')
        target.write_bytes(payload)
        entries.append({'name': name, 'repo': repo, 'revision': revision, 'source_license': expected,
                        'source_url': f'https://github.com/{repo}/blob/{revision}/LICENSE',
                        'evidence': str(target.relative_to(ROOT)), 'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)})
    report = {'entries': entries,
              'scope': 'Exact pinned dataset/model card license declarations and two source LICENSE files; not proof of training exposure, ownership of all underlying material, or identical code/weights licensing.',
              'remaining_release_work': ['Preserve complete license/NOTICE texts and dependent base-model terms in any distributed weight package.', 'Choose and record a license for original project code; third-party licenses remain separate.', 'Check anonymized release bundle against ARR paper/source revision after final experiments.'],
              'openjev_weights': 'Uses the pinned Qwen3-0.6B card above; the MIT entry describes upstream OpenJev source only.'}
    (ROOT / 'handoff/release-provenance.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = ['# 固定版本来源与许可声明', '', '以下为逐版本卡片声明和源码 LICENSE 的证据记录，完整 SHA 见同名 JSON。模型卡声明与源码许可分开记录；这些证据不证明训练材料不存在重叠。', '', '| 项目 | 固定版本 | 卡片／源码声明 | 证据 |', '|---|---|---|---|']
    for entry in entries:
        declaration = entry.get('card_declared_license', entry.get('source_license'))
        lines.append(f"| {entry['name']} | `{entry['revision']}` | {declaration} | [{entry['evidence']}]({entry['source_url']}) |")
    lines += ['', 'OpenJev源码为MIT；其实际载入的Qwen权重使用Qwen固定版本的卡片声明，不把源码MIT自动套用到权重。', '', '本地BANKING77处理派生了训练／开发／校准／候选列表；CLINC使用plus配置并派生去重分割和oracle八候选题。论文和复现材料应保留原作者引用、这些变换说明及来源。合成policy来自项目的确定性规则生成器，不冒称真实业务数据。', '', '发布权重包前仍需整理完整许可证／NOTICE及依赖基础模型条款；项目原创代码尚未选定许可。最终匿名复现包须与最终论文版本对应。本文记录不表示已完成发布包审核。']
    (ROOT / 'handoff/RELEASE_PROVENANCE.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'verified_entries': len(entries), 'report': 'handoff/release-provenance.json'}))


if __name__ == '__main__':
    main()
