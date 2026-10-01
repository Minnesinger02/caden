"""JSONL inference CLI; no benchmark labels or metrics are needed."""
import argparse
import json
from pathlib import Path
import sys

import torch

from .api import Caden, MODEL_ID
from .data import read_jsonl


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID, help="Hub repository or local checkpoint/bundle")
    parser.add_argument("--revision", help="Hub commit; default Caden release is pinned")
    parser.add_argument("--seed", type=int, choices=[42, 43, 44], default=42)
    parser.add_argument("--input", required=True, help="JSONL requests")
    parser.add_argument("--output", help="New JSONL output; defaults to stdout")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--temperature", type=float, help="Override checkpoint calibration; use 1 for raw")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.cpu_threads < 1:
        parser.error("batch-size and cpu-threads must be positive")
    if args.output and Path(args.output).exists():
        parser.error("Output already exists; use a new file")
    torch.set_num_threads(args.cpu_threads)
    rows = read_jsonl(args.input)
    model = Caden.from_pretrained(args.model, args.revision, args.seed,
                                 args.device, args.temperature)
    answers = model.predict_many(rows, args.batch_size)
    stream = Path(args.output).open("x", encoding="utf-8") if args.output else sys.stdout
    try:
        for answer in answers:
            print(json.dumps(answer, ensure_ascii=False), file=stream)
    finally:
        if args.output:
            stream.close()


if __name__ == "__main__":
    main()
