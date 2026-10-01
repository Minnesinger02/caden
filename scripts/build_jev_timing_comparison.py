"""Compare saved application latencies without additional paid requests."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'results/jev113-budgeted-v2/banking-dev'
    predictions = [json.loads(line) for line in (source / 'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    metadata = json.loads((source / 'metadata.json').read_text(encoding='utf-8'))
    api = json.loads((source / 'metrics.json').read_text(encoding='utf-8'))
    local_path = ROOT / 'results/replicated-task-fp32-8792/summary-mixed-reference.json'
    local = json.loads(local_path.read_text(encoding='utf-8'))
    if len(predictions) != 200 or api['failures'] or metadata['requested_model'] != 'jev-1.13.0':
        raise ValueError('Complete pinned same-task API timing required')
    rows = []
    for item in local['rows']:
        run = ROOT / ('results/replicated-task-fp32-8792/block0-' + item['model'])
        run_meta = json.loads((run / 'metadata.json').read_text(encoding='utf-8'))
        if run_meta['dataset_sha256'] != metadata['dataset_sha256']:
            raise ValueError('Timing task bytes differ')
        rows.append({'system': item['model'], 'scope': 'local GPU FP32 B1, five serial randomized blocks',
            'p50_ms': statistics.median(item['p50_ms_by_block']),
            'latency_rate_per_second': item['median_decisions_per_second']})
    summed_seconds = sum(row['latency_ms'] for row in predictions) / 1000
    api_row = {'system': 'jev-1.13.0', 'scope': 'remote HTTP, concurrency4, one pass, no warmup; server precision/hardware unknown',
        'p50_ms': api['latency_p50_ms'], 'p95_ms': api['latency_p95_ms'],
        'mean_ms': summed_seconds * 1000 / 200,
        'latency_rate_per_second': 200 / summed_seconds,
        'request_latency_sum_seconds': summed_seconds, 'items': 200}
    rows.append(api_row)
    paths = [source / 'predictions.jsonl', source / 'metadata.json', source / 'metrics.json', local_path]
    report = {'dataset_sha256': metadata['dataset_sha256'], 'items': 200, 'rows': rows,
        'source_files': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'definition': 'Latency rate = questions / summed per-request latency, not inverse p50. Local reports the median rate over five blocks. API has one pass at HTTP concurrency4, so its latency rate is neither observed aggregate concurrent throughput nor a serial measured rate. Concurrent request overlap is retained. No cross-runtime paired GPU ratio or API timing CI is claimed.'}
    (ROOT / 'results/jev-local-timing-comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = ['# Jev与本地应用时延对比', '', report['definition'], '',
        '| 系统 | p50/ms | 题数/请求时延和（每秒） | 测量条件 |', '| --- | ---: | ---: | --- |']
    for row in rows:
        lines.append(f"| {row['system']} | {row['p50_ms']:.2f} | {row['latency_rate_per_second']:.2f} | {row['scope']} |")
    lines += ['', 'API不新增调用；所有系统使用相同200条BANKING开发payload。四并发HTTP时延含网络／服务端，表中不能当同卡GPU加速或API并发总吞吐量。']
    (ROOT / 'results/jev-local-timing-comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(api_row, indent=2))


if __name__ == '__main__':
    main()
