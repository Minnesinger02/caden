"""Inventory recorded local training costs without inferring active GPU hours."""
import hashlib
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    loops = []
    for path in sorted((ROOT / 'checkpoints').rglob('*.json')):
        if path.name not in {'training.json', 'experiment.json'}:
            continue
        record = json.loads(path.read_text(encoding='utf-8'))
        if record.get('command') != 'train' or record.get('device') != 'cuda':
            continue
        data = ROOT / record['data'].replace('\\', '/')
        expected = record.get('data_sha256', record.get('dataset_sha256'))
        if not expected or sha(data) != expected:
            raise ValueError(f'Training input changed: {path}')
        seconds = record['training_seconds']
        if not math.isfinite(seconds) or seconds <= 0 or record['microsteps'] <= 0:
            raise ValueError('Invalid training timing record')
        loops.append({'record': path.relative_to(ROOT).as_posix(), 'sha256': sha(path),
            'checkpoint': path.parent.relative_to(ROOT).as_posix(), 'seed': record['seed'],
            'model': record['model'], 'data_sha256': expected, 'epochs': record['epochs'],
            'microsteps': record['microsteps'], 'recorded_training_loop_seconds': seconds,
            'trainable_parameters_recorded': record.get('trainable_parameters'),
            'timing_scope': record.get('timing_scope'),
            'known_discontinuity': path.parent.name == 'encoder16-s44-e3-mixed23789'})
    if len(loops) != 32:
        raise ValueError('Current registered checkpoint inventory changed; review scope')
    stages = []
    for path in sorted((ROOT / 'handoff/runs').rglob('stage.json')):
        record = json.loads(path.read_text(encoding='utf-8'))
        if record.get('stage') not in {'encoder-train', 'sft-train'}:
            continue
        if record.get('exit_code') is None:
            raise ValueError('Training stage lacks terminal state')
        stages.append({'record': path.relative_to(ROOT).as_posix(), 'sha256': sha(path),
            'stage': record['stage'], 'command': record.get('command'), 'exit_code': record['exit_code'],
            'recorded_stage_wall_seconds': record.get('wall_seconds'),
            'board_peak_memory_mib': record.get('board_peak_memory_mib')})
    tensors = []
    for path in sorted((ROOT / '.cache/community').rglob('*.safetensors')):
        with path.open('rb') as stream:
            size = struct.unpack('<Q', stream.read(8))[0]
            if size > 32 * 1024 * 1024:
                raise ValueError('Unexpected tensor header length')
            header_bytes = stream.read(size)
            header = json.loads(header_bytes)
        values = [value for key, value in header.items() if key != '__metadata__']
        tensors.append({'file': path.relative_to(ROOT).as_posix(), 'header_sha256': hashlib.sha256(header_bytes).hexdigest(),
            'file_bytes': path.stat().st_size, 'tensor_count': len(values),
            'serialized_tensor_elements': sum(math.prod(value['shape']) for value in values),
            'dtype_counts': {dtype: sum(value['dtype'] == dtype for value in values) for dtype in sorted({value['dtype'] for value in values})}})
    loop_sum = sum(row['recorded_training_loop_seconds'] for row in loops)
    wall_sum = sum(row['recorded_stage_wall_seconds'] or 0 for row in stages)
    report = {'training_records': loops, 'recorded_training_loop_hours': loop_sum / 3600,
        'training_stage_records': stages, 'recorded_training_stage_wall_hours': wall_sum / 3600,
        'training_stage_exit_counts': {str(code): sum(row['exit_code'] == code for row in stages) for code in sorted({row['exit_code'] for row in stages})},
        'serialized_tensor_inventory': tensors,
        'known_gap_report': {'path': 'handoff/encoder44-training-timing-gap.json',
                             'sha256': sha(ROOT / 'handoff/encoder44-training-timing-gap.json')},
        'scope': '32 saved CUDA training-loop records, plus all discovered handoff/runs training stage records including failures. These scopes overlap and must NOT be added together. Stage wall includes loading/validation and recorded pauses; loop timers exclude loading/saving. Neither measures active GPU compute. Inference, tests, downloads, unlogged attempts and pretraining are excluded; this is not total project GPU hours. Serialized tensor elements are storage inventory, not certified model parameter counts (buffers/tied tensors/heads may differ). No GPU model loaded.',
        'total_project_active_gpu_hours_verified': False}
    target = ROOT / 'handoff/RECORDED_TRAINING_BUDGET.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 实际记录的训练成本', '',
        f'已保存{len(loops)}组CUDA训练循环，总记录时间{loop_sum / 3600:.4f}小时。',
        f'训练stage记录{len(stages)}组，退出码计数{report["training_stage_exit_counts"]}，已记录stage墙钟合计{wall_sum / 3600:.4f}小时。', '',
        '两个统计范围重合，不相加；不是完整项目GPU小时。loop排除加载／保存，stage含加载等开销与记录到的暂停；包含seed44已知数小时计时缺口，不减去猜测的暂停。未覆盖推理、测试、下载、未记录尝试和上游预训练，不能用这些总数比较不间断训练速度。', '',
        '| 保存的训练循环 | 种子 | 轮数 | microsteps | 记录秒数 | 已知缺口 |',
        '| --- | ---: | ---: | ---: | ---: | --- |']
    for row in loops:
        lines.append(f'| {row["checkpoint"]} | {row["seed"]} | {row["epochs"]} | {row["microsteps"]} | {row["recorded_training_loop_seconds"]:.2f} | {row["known_discontinuity"]} |')
    lines += ['', '## 社区tensor存储清单', '',
              '读取safetensors头部，不加载GPU；下列元素数量不是经过模型实例验证的参数量，不能直接替换论文参数数。', '',
              '| 文件 | 存储tensor数 | 元素数 |', '| --- | ---: | ---: |']
    for row in tensors:
        lines.append(f'| {row["file"]} | {row["tensor_count"]} | {row["serialized_tensor_elements"]} |')
    (ROOT / 'handoff/RECORDED_TRAINING_BUDGET.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ['recorded_training_loop_hours', 'recorded_training_stage_wall_hours', 'training_stage_exit_counts', 'total_project_active_gpu_hours_verified']}, indent=2))


if __name__ == '__main__':
    main()
