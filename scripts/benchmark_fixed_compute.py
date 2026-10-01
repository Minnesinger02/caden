"""Fixed GPU/FP32/token-shape throughput; decisions and generation are distinct."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
import torch
from transformers import AutoModelForCausalLM
from decision_lab.encoder import CandidateEncoder

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['encoder', 'qwen'], required=True)
    parser.add_argument('--checkpoint', default='checkpoints/encoder16-s42-e3-5k')
    parser.add_argument('--lengths', nargs='+', type=int, default=[128, 256])
    parser.add_argument('--batches', nargs='+', type=int, default=[1, 8, 16])
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--include-generation', action='store_true', help='Optional historical generation arm; decision-only is the default')
    parser.add_argument('--dtype', choices=['float32','bfloat16'], default='float32')
    parser.add_argument('--attention-impl', choices=['eager','sdpa'], default='eager')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if Path(args.out).exists() or min(args.lengths + args.batches + [args.repeats]) < 1:
        parser.error('Use fresh output and positive dimensions')
    if min(args.lengths) < 8 or max(args.lengths) > 512:
        parser.error('Lengths must be 8..512 for this shared shape benchmark')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    dtype = getattr(torch, args.dtype)
    lock = json.loads((ROOT / 'configs/resolved_revisions-parquet.json').read_text(encoding='utf-8'))
    if args.backend == 'encoder':
        model = CandidateEncoder.load(args.checkpoint, args.attention_impl).to(device='cuda', dtype=dtype).eval()
        if model.input_mode != 'joint' or model.readout != 'scalar':
            raise ValueError('This benchmark implements joint scalar readout only; independent/pointer compute requires its own workload')
        revision = json.loads((Path(args.checkpoint) / 'training.json').read_text(encoding='utf-8'))
    else:
        model = AutoModelForCausalLM.from_pretrained(lock['decoder']['id'], revision=lock['decoder']['revision'],
                                                   torch_dtype=dtype, attn_implementation=args.attention_impl).to('cuda').eval()
        revision = lock['decoder']
    rows = []
    with torch.inference_mode():
        for length in args.lengths:
            for batch in args.batches:
                ids = torch.randint(500, 20000, (batch, length), device='cuda')
                mask = torch.ones_like(ids)
                def score():
                    if args.backend == 'encoder':
                        hidden = model.backbone(input_ids=ids, attention_mask=mask).last_hidden_state
                        return model.head(hidden[:, -8:]).squeeze(-1).float().softmax(-1)
                    hidden = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state[:, -1]
                    return torch.nn.functional.linear(hidden, model.get_output_embeddings().weight[32:40]).float().softmax(-1)
                for _ in range(5):
                    score()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                for _ in range(args.repeats):
                    score()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                rows.append({'task': '8-candidate probability readout', 'batch': batch, 'input_tokens_per_item': length,
                             'iterations': args.repeats, 'seconds': elapsed,
                             'decisions_per_second': batch * args.repeats / elapsed,
                             'input_tokens_per_second': batch * length * args.repeats / elapsed,
                             'generated_tokens_per_second': None, 'peak_allocated_bytes': torch.cuda.max_memory_allocated()})
                if args.include_generation and args.backend == 'qwen' and batch in [1, 8]:
                    def generate():
                        return model.generate(input_ids=ids, attention_mask=mask, min_new_tokens=64, max_new_tokens=64,
                                              do_sample=False, use_cache=True, pad_token_id=0)
                    generate()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    count = 0
                    for _ in range(3):
                        result = generate()
                        count += result.shape[0] * (result.shape[1] - length)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    assert count == batch * 64 * 3
                    rows.append({'task': '64-token greedy generation with KV cache', 'batch': batch,
                                 'input_tokens_per_item': length, 'generated_tokens': count, 'seconds': elapsed,
                                 'generated_tokens_per_second': count / elapsed,
                                 'scope': 'includes prefill and all 64 decode tokens; full vocabulary head; no tokenization/readback',
                                 'peak_allocated_bytes': torch.cuda.max_memory_allocated()})
    report = {'settings': vars(args), 'model_metadata': revision, 'gpu': torch.cuda.get_device_name(),
              'torch': torch.__version__, 'dtype': args.dtype, 'attention': args.attention_impl, 'rows': rows,
              'scope': 'synthetic nonpadding IDs, identical B x L dimensions, GPU-resident input/output, warmup5; probability output is not generated tokens; architecture and parameter sizes differ'}
    with Path(args.out).open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(rows, indent=2))

if __name__ == '__main__':
    main()
